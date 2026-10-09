"""Run, save, verify and display the two CatalogIQ splitting experiments."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path, PureWindowsPath
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from .cleaning import PROVENANCE
from .features import sha256
from .splitting import (
    IDENTIFIER_FIELDS, SplitConfig, check_assignments, record_ids, rule_groups,
    stratified_group_split, tfidf_groups,
)

VERSION = "split-comparison-v1"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_input(input_path: Path, identifier_path: Path | None = None):
    """Read retained training records and optionally recover audited identifiers.

    The default export manifest points to the completed integrated candidate
    artifact. Identity joins never add/drop rows or use the prediction dataset.
    """
    input_path = Path(input_path).resolve()
    hashes = {str(input_path): sha256(input_path)}
    manifest_path = input_path.with_suffix(".summary.json")
    export = None
    if manifest_path.exists():
        export = json.loads(manifest_path.read_text(encoding="utf-8"))
        if export.get("output_sha256") != hashes[str(input_path)]:
            raise ValueError("Cleaned input differs from its export manifest")
        hashes[str(manifest_path)] = sha256(manifest_path)
        if identifier_path is None and "source_run" in export:
            run_dir = Path(export["source_run"])
            if not run_dir.exists():
                # A copied checkout can retain an absolute original run path.
                original = export["source_run"]
                run_name = PureWindowsPath(original).name if "\\" in original else run_dir.name
                run_dir = input_path.parent / run_name
            identifier_path = run_dir / "training_candidate.csv"
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if frame.empty or "Segment" not in frame.columns:
        raise ValueError("Input must contain records and the Segment target")
    ids = record_ids(frame)
    if export and len(frame) != export["rows"]:
        raise ValueError("Input row count differs from its export manifest")
    added = []
    if identifier_path is not None:
        identifier_path = Path(identifier_path).resolve()
        digest = sha256(identifier_path)
        hashes[str(identifier_path)] = digest
        integration_manifest = identifier_path.parent / "summary.json"
        if integration_manifest.exists():
            integrated = json.loads(integration_manifest.read_text(encoding="utf-8"))
            expected = integrated.get("output_sha256", {}).get(identifier_path.name)
            if expected is not None and expected != digest:
                raise ValueError("Identifier artifact differs from its integrated manifest")
            hashes[str(integration_manifest)] = sha256(integration_manifest)
        if (export and identifier_path.name in export.get("input_sha256", {})
                and export["input_sha256"][identifier_path.name] != digest):
            raise ValueError("Identifier artifact differs from the cleaned export's source")
        other = pd.read_csv(identifier_path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                            usecols=lambda c: c in (*PROVENANCE, *IDENTIFIER_FIELDS))
        record_ids(other)  # Validate source identity before joining.
        existing = [c for c in IDENTIFIER_FIELDS if c in frame and c in other]
        joined = frame[PROVENANCE + existing].merge(other, on=PROVENANCE, how="left",
                                                  validate="one_to_one", suffixes=("", "_audit"),
                                                  indicator=True, sort=False)
        if not joined["_merge"].eq("both").all():
            raise ValueError("Identifier artifact does not cover every retained record")
        for c in existing:
            if not joined[c].equals(joined[c + "_audit"]):
                raise ValueError(f"Conflicting identifier values in {c}")
        for c in IDENTIFIER_FIELDS:
            if c not in frame and c in joined:
                frame[c] = joined[c].to_numpy()
                added.append(c)
    frame["record_id"] = ids
    frame = frame.sort_values("record_id", kind="stable").reset_index(drop=True)
    metadata = {
        "path": str(input_path), "sha256": hashes[str(input_path)], "rows": len(frame),
        "identifier_source": str(identifier_path) if identifier_path else None,
        "joined_identifier_columns": added, "fingerprints": hashes,
        "target": "Segment", "eligibility": "all retained cleaned records; missing Segment retained as <MISSING>",
        "prediction_target_records_used": False,
    }
    return frame, metadata


def attribute_profile(frame):
    from .splitting import GROUP_FIELDS, gtin, text
    rows = []
    for column in GROUP_FIELDS:
        if column not in frame:
            continue
        values = frame[column].map(text)
        populated = values[values.ne("")]
        frequencies = populated.value_counts()
        rows.append({"column": column, "populated_rows": len(populated),
                     "distinct_populated": len(frequencies),
                     "repeated_values": int(frequencies.gt(1).sum()),
                     "largest_value_count": int(frequencies.max()) if len(frequencies) else 0})
    return pd.DataFrame(rows), {
        "checksum_valid_gtin_rows": int(frame.get("Upc", pd.Series(dtype=str)).map(gtin).ne("").sum()),
        "excluded_identity_fields": ["MDM_Id", "JoiningKey"],
        "identifier_policy": "intact checksum-valid GTIN; retailer+SKU and brand+model require name agreement; MDM/JoiningKey are traceability only",
        "rule_blocking": "brand plus infrequent name tokens; oversized fuzzy blocks skipped and counted",
        "grouping_label_columns_used": [],
    }


def _assign(frame, groups, config):
    splits, stats = stratified_group_split(groups, frame["Segment"], config)
    assignment = frame[PROVENANCE + ["record_id", "Segment"]].copy()
    assignment["group_id"] = groups
    assignment["split"] = splits
    check_assignments(assignment, frame["record_id"])
    return assignment, stats


def _edges_frame(edges, frame):
    return pd.DataFrame([{"left_record_id": frame.at[a, "record_id"],
                          "right_record_id": frame.at[b, "record_id"],
                          "reason": reason, "score": score} for a, b, reason, score in edges],
                        columns=["left_record_id", "right_record_id", "reason", "score"])


def disagreement_pairs(frame, assignments, edges, evaluation_pairs, limit):
    """Count partition disagreements exactly; show representative witnesses.

    Counts include all within-component pairs, including transitive links.
    Examples draw from matching edges and independent evaluation pairs.
    """
    def choose_two(counts):
        return int(sum(int(n) * (int(n) - 1) // 2 for n in counts))

    rule = assignments["rule"]["group_id"].to_numpy()
    tfidf = assignments["tfidf"]["group_id"].to_numpy()
    intersections = pd.DataFrame({"rule": rule, "tfidf": tfidf}).value_counts()
    both = choose_two(intersections)
    only_rule = choose_two(Counter(rule).values()) - both
    only_tfidf = choose_two(Counter(tfidf).values()) - both
    candidates = {}
    for method, links in edges.items():
        for a, b, reason, score in links:
            key = tuple(sorted((a, b)))
            candidates.setdefault(key, (reason, score, method))
    for a, b, score in evaluation_pairs:
        candidates.setdefault((a, b), ("independent_name_jaccard", score, "evaluation"))
    examples = {"rule_only": [], "tfidf_only": []}
    # Stable high-score ordering makes saved examples reproducible.
    for (a, b), (reason, score, origin) in sorted(candidates.items(), key=lambda item: (-item[1][1], item[0])):
        same_rule, same_tfidf = rule[a] == rule[b], tfidf[a] == tfidf[b]
        if same_rule == same_tfidf:
            continue
        direction = "rule_only" if same_rule else "tfidf_only"
        if len(examples[direction]) >= limit:
            continue
        row = {"direction": direction, "witness": reason, "score": score, "origin": origin,
               "same_rule_group": bool(same_rule), "same_tfidf_group": bool(same_tfidf)}
        for side, i in (("left", a), ("right", b)):
            for c in ("record_id", *PROVENANCE, "ProductName", "ProductBrand", "Retailer", "Sku", "Upc",
                      "ProductModelNumber", "ProductUrl", "ProductDescription", "Segment"):
                row[f"{side}_{c}"] = frame.at[i, c] if c in frame else ""
            row[f"{side}_rule_group"] = rule[i]
            row[f"{side}_tfidf_group"] = tfidf[i]
            row[f"{side}_rule_split"] = assignments["rule"].at[i, "split"]
            row[f"{side}_tfidf_split"] = assignments["tfidf"].at[i, "split"]
        examples[direction].append(row)
    rows = examples["rule_only"] + examples["tfidf_only"]
    example_frame = pd.DataFrame(rows)
    if not rows:
        example_frame = pd.DataFrame(columns=["direction", "witness", "score", "origin",
            "same_rule_group", "same_tfidf_group", "left_ProductName", "right_ProductName"])
    return example_frame, {
        "co_grouped_pairs_both": both, "co_grouped_pairs_rule_only": only_rule,
        "co_grouped_pairs_tfidf_only": only_tfidf,
        "example_sampling": "highest-score direct matching/evaluation witnesses, up to configured limit per direction",
    }


def _environment():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
            "git_revision": revision, "git_dirty": dirty,
            "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("split*.py")}}


def run_experiment(input_path, output_dir, config=None, identifier_path=None,
                   verify_reproducibility=True, progress=print):
    from .split_evaluation import leakage_metrics, near_duplicate_pairs, segment_distribution
    config = config or SplitConfig()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; use a new run directory")
    started = perf_counter()
    frame, input_metadata = load_input(Path(input_path), identifier_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", config.to_dict())
    _write_json(output_dir / "input_manifest.json", input_metadata)
    profile, identity_policy = attribute_profile(frame)
    if config.grouping_version == 2:
        identity_policy.update({
            "identifier_policy": "v2: checksum-valid GTIN, listing URL, retailer+SKU and brand+model all require name/brand/core/formulation corroboration; MDM/JoiningKey are traceability only",
            "rule_blocking": "v2: compact canonical brand and title-prefix blocks plus infrequent core tokens; guarded exact-name/family passes; oversized fuzzy blocks skipped and counted",
            "formulation_policy": "disjoint known active/compound signatures reject links, including links between components; missing signatures are not evidence of equivalence",
        })
    profile.to_csv(output_dir / "attribute_profile.csv", index=False)
    assignments, edges, method_stats = {}, {}, {}
    for method, grouper in (("rule", rule_groups), ("tfidf", tfidf_groups)):
        progress(f"Grouping {len(frame):,} retained records: {method}")
        start = perf_counter()
        kwargs = {"progress": progress} if method == "tfidf" else {}
        groups, links, stats = grouper(frame, frame["record_id"].to_numpy(), config, **kwargs)
        grouping_seconds = perf_counter() - start
        start = perf_counter()
        assignment, split_stats = _assign(frame, groups, config)
        splitting_seconds = perf_counter() - start
        assignments[method], edges[method] = assignment, links
        assignment.to_csv(output_dir / f"{method}_assignments.csv", index=False, lineterminator="\n")
        _edges_frame(links, frame).to_csv(output_dir / f"{method}_matching_edges.csv", index=False)
        group_sizes = assignment.groupby("group_id", sort=True).size().rename("rows").reset_index()
        group_sizes.to_csv(output_dir / f"{method}_group_sizes.csv", index=False)
        group_sizes["rows"].value_counts().sort_index().rename_axis("group_size").rename("groups").to_csv(
            output_dir / f"{method}_group_size_distribution.csv")
        method_stats[method] = {"grouping": stats, "split_allocator": split_stats,
                                "runtime_seconds": {"grouping": grouping_seconds,
                                                    "splitting": splitting_seconds,
                                                    "primary_total": grouping_seconds + splitting_seconds}}
        progress(f"{method}: {len(group_sizes):,} groups, {len(links):,} accepted links; "
                 f"grouping {grouping_seconds:.2f}s, splitting {splitting_seconds:.2f}s")
    progress("Evaluating both assignments with shared independent character-shingle Jaccard pairs")
    start = perf_counter()
    evaluation_pairs, evaluation_stats = near_duplicate_pairs(frame, config, progress=progress)
    evaluation_stats["runtime_seconds"] = perf_counter() - start
    pair_rows = []
    for a, b, score in evaluation_pairs:
        row = {"left_record_id": frame.at[a, "record_id"], "right_record_id": frame.at[b, "record_id"],
               "name_jaccard": score}
        for method in assignments:
            row[f"{method}_left_split"] = assignments[method].at[a, "split"]
            row[f"{method}_right_split"] = assignments[method].at[b, "split"]
            row[f"{method}_crosses_split"] = row[f"{method}_left_split"] != row[f"{method}_right_split"]
        pair_rows.append(row)
    pd.DataFrame(pair_rows, columns=["left_record_id", "right_record_id", "name_jaccard",
        "rule_left_split", "rule_right_split", "rule_crosses_split",
        "tfidf_left_split", "tfidf_right_split", "tfidf_crosses_split"]).to_csv(
        output_dir / "independent_near_duplicate_pairs.csv", index=False)
    distributions = []
    for method, assignment in assignments.items():
        method_stats[method]["metrics"] = leakage_metrics(frame, assignment, evaluation_pairs, config)
        distribution = segment_distribution(frame, assignment)
        distribution.insert(0, "approach", method)
        distributions.append(distribution)
    pd.concat(distributions, ignore_index=True).to_csv(output_dir / "segment_distributions.csv", index=False)
    examples, disagreement_stats = disagreement_pairs(frame, assignments, edges, evaluation_pairs,
                                                       config.disagreement_examples)
    examples.to_csv(output_dir / "disagreement_examples.csv", index=False)
    checks = {"same_records": True, "complete_unique_coverage": True, "group_isolation": True,
              "labels_excluded_from_grouping": True, "reproducibility": {}}
    if verify_reproducibility:
        progress("Checking reproducibility: independently rebuilding both groupings and allocations")
        for method, grouper in (("rule", rule_groups), ("tfidf", tfidf_groups)):
            start = perf_counter()
            kwargs = {"progress": progress} if method == "tfidf" else {}
            groups, _, _ = grouper(frame, frame["record_id"].to_numpy(), config, **kwargs)
            repeated, _ = _assign(frame, groups, config)
            if not assignments[method].equals(repeated):
                raise RuntimeError(f"{method} grouping/splitting is not reproducible")
            checks["reproducibility"][method] = {"passed": True, "runtime_seconds": perf_counter() - start}
    else:
        checks["reproducibility"] = {"status": "not run; explicitly disabled"}
    for name, expected in input_metadata["fingerprints"].items():
        if sha256(Path(name)) != expected:
            raise ValueError("Input artifact changed during the experiment")
    checks["input_artifacts_unchanged"] = True
    summary = {"version": VERSION if config.grouping_version == 1 else "split-comparison-v2", "created_utc": datetime.now(timezone.utc).isoformat(),
               "input": input_metadata, "config": config.to_dict(), "environment": _environment(),
               "identity_policy": identity_policy,
               "group_policy": "connected components; transitive chains may contain dissimilar endpoints",
               "tfidf_policy": ("v2: uncapped full-title and packaging-family unigram/bigram TF-IDF threshold graphs; descriptions/contents support borderline title matches and cannot weaken strong titles; shared brand/core/formulation and component conflict guards; title/brand weights unused, support weights retained; IDF fit on unlabeled splitting universe only"
                   if config.grouping_version == 2 else "v1: separate word unigram/bigram TF-IDF per text field; weighted and L2 normalized; full threshold graph in sparse chunks; IDF fit on unlabeled splitting universe only"),
               "split_policy": "same seeded whole-group optimizer, per-class and total-size squared allocation errors; labels used only after grouping",
               "evaluation": evaluation_stats, "approaches": method_stats,
               "disagreements": disagreement_stats, "checks": checks,
               "total_wall_seconds_including_io_evaluation_and_reproduction": perf_counter() - started}
    comparison_table(summary).to_csv(output_dir / "comparison.csv", index_label="metric")
    summary["output_sha256"] = {p.name: sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()}
    _write_json(output_dir / "summary.json", summary)
    display_results(output_dir)
    return summary


def verify_saved(output_dir):
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    for name, digest in summary["output_sha256"].items():
        if sha256(output_dir / name) != digest:
            raise ValueError(f"Saved artifact changed: {name}")
    for name, digest in summary["input"]["fingerprints"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Original input artifact changed: {name}")
    frame, _ = load_input(Path(summary["input"]["path"]),
                          Path(summary["input"]["identifier_source"]) if summary["input"]["identifier_source"] else None)
    if sha256(Path(summary["input"]["path"])) != summary["input"]["sha256"]:
        raise ValueError("Shared input fingerprint changed")
    config = SplitConfig.from_dict(summary["config"])
    for method in ("rule", "tfidf"):
        assignment = pd.read_csv(output_dir / f"{method}_assignments.csv", dtype=str, keep_default_na=False)
        check_assignments(assignment, frame["record_id"])
        assignment = assignment.set_index("record_id").loc[frame["record_id"]].reset_index()
        # Full grouping rebuild was recorded by the experiment; this independent
        # command validates artifacts and reproduces allocation from saved groups.
        repeated, _ = _assign(frame, assignment["group_id"].to_numpy(), config)
        if not assignment[repeated.columns].equals(repeated):
            raise ValueError(f"Saved {method} allocation does not reproduce")
    return {"artifact_hashes": True, "record_coverage": True, "group_isolation": True,
            "allocation_reproduced": True}


def comparison_table(summary):
    rows = {}
    for method, data in summary["approaches"].items():
        m = data["metrics"]
        row = {
            "records": m["rows"], "groups": m["groups"]["count"],
            "singleton groups": m["groups"]["singleton_groups"],
            "mean group size": m["groups"]["size_summary"]["mean"],
            "median group size": m["groups"]["size_summary"]["0.5"],
            "p95 group size": m["groups"]["size_summary"]["0.95"],
            "largest group": m["groups"]["size_summary"]["1.0"],
            "mixed Segment groups": m["groups"]["mixed_segment_groups"],
        }
        for split in ("train", "validation", "test"):
            row[f"{split} rows"] = m["splits"][split]["rows"]
            row[f"{split} proportion"] = m["splits"][split]["proportion"]
        row.update({
            "exact payload duplicate pairs": m["exact_payload"]["duplicate_pairs"],
            "exact payload crossing pairs": m["exact_payload"]["crossing_pairs"],
            "normalized brand/name crossing pairs": m["exact_brand_name"]["crossing_pairs"],
            "independent near duplicate pairs": m["near_duplicate"]["pairs"],
            "near duplicate crossing pairs": m["near_duplicate"]["crossing_pairs"],
            "near duplicate crossing pair rate": m["near_duplicate"]["crossing_pair_rate"],
            "near duplicate crossing rows": m["near_duplicate"]["crossing_rows"],
            "nonidentical name crossing pairs": m["near_duplicate"]["nonidentical_name_crossing_pairs"],
            "missing validation Segment classes": len(m["missing_segment_classes"]["validation"]),
            "missing test Segment classes": len(m["missing_segment_classes"]["test"]),
            "rare classes missing validation": len(m["rare_segment_classes_missing_validation"]),
            "rare classes missing test": len(m["rare_segment_classes_missing_test"]),
            "grouping seconds": data["runtime_seconds"]["grouping"],
            "splitting seconds": data["runtime_seconds"]["splitting"],
            "primary total seconds": data["runtime_seconds"]["primary_total"],
        })
        if "co_grouped_pairs" in m["near_duplicate"]:
            row["independent near duplicate co-grouped pairs"] = m["near_duplicate"]["co_grouped_pairs"]
            row["independent near duplicate co-grouped rate"] = m["near_duplicate"]["co_grouped_pair_rate"]
        rows[method] = row
    return pd.DataFrame(rows)


def display_results(output_dir):
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    print("\nSide-by-side split comparison (shared input and shared evaluator):")
    print(comparison_table(summary).to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nGroup size distribution (number of groups):")
    print(pd.DataFrame({method: data["metrics"]["groups"]["size_histogram"]
                        for method, data in summary["approaches"].items()}).to_string())
    print("\nSegment counts and proportions:")
    distribution = pd.read_csv(output_dir / "segment_distributions.csv", keep_default_na=False)
    columns = [f"{split}_rows" for split in ("train", "validation", "test")]
    print(distribution.pivot(index="Segment", columns="approach", values=columns).to_string())
    columns = [f"{split}_class_proportion" for split in ("train", "validation", "test")]
    print(distribution.pivot(index="Segment", columns="approach", values=columns).to_string(
        float_format=lambda x: f"{x:.5f}"))
    print("\nMissing Segment classes:")
    print(json.dumps({method: data["metrics"]["missing_segment_classes"]
                      for method, data in summary["approaches"].items()}, indent=2))
    print("\nPartition disagreements:")
    print(json.dumps(summary["disagreements"], indent=2))
    examples = pd.read_csv(output_dir / "disagreement_examples.csv", keep_default_na=False)
    if len(examples):
        print("\nSample disagreements (full examples in disagreement_examples.csv):")
        for direction in ("rule_only", "tfidf_only"):
            selected = examples[examples["direction"].eq(direction)].head(2)
            for _, row in selected.iterrows():
                print(f"{direction} [{row['witness']}, score {row['score']:.4f}]")
                print(f"  {row['left_ProductName'][:180]}")
                print(f"  {row['right_ProductName'][:180]}")
    print(f"\nSaved artifacts: {output_dir}")
