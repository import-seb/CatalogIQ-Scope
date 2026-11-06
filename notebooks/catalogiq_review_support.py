"""Read-only presentation helpers for the 2026-10-30 CatalogIQ sprint review.

No training, model loading, inference, source-data reads, or test-label access.
Only explicit development files and existing validation exports are consumed.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, SVG, display
from sklearn.metrics import accuracy_score, classification_report, f1_score


EXPORT = "data/processed/minilm_validation_exports_v1"
SPLIT = "data/processed/split_integration_reproduced_20261010"
CLEAN = "data/processed/integrated_policy_v4_final_20261003"
REMAINING = "data/processed/minilm_remaining_targets_v1_20261029"
CATEGORY = "data/processed/segment_category_ablation_256_20261010"
TRANSFER = "data/processed/segment_retailer_holdout_category_20261010"
TARGETS = {
    "segment": ("Segment", 11952, 8),
    "mnfr": ("Mnfr", 11812, 2),
    "targetagegroup": ("TargetAgeGroup", 11624, 3),
    "brand": ("Brand", 12619, 143),
    "sub_segment": ("Sub-Segment", 11562, 49),
}
ALLOCATIONS = {
    "Brand": "data/processed/brand_development_allocation_v1_20261029",
    "Sub-Segment": "data/processed/sub_segment_development_allocation_v1_20261029_r2",
}
NAVY, TEAL, GOLD, GREY = "#18324d", "#087f8c", "#c96b30", "#718096"

STYLE = """
<style>
.jp-Notebook, .jp-RenderedHTMLCommon {--jp-content-font-size1:16px;}
.jp-Notebook {max-width:1180px;margin:auto;}
.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2 {color:#18324d;letter-spacing:-.025em;}
.jp-RenderedHTMLCommon h1 {font-size:2.8em;line-height:1.12;}
.jp-RenderedHTMLCommon h2 {font-size:1.9em;border-top:3px solid #087f8c;padding-top:18px;}
.jp-RenderedHTMLCommon p,.jp-RenderedHTMLCommon li {line-height:1.55;}
.cq-kicker {color:#087f8c;font-size:12px;font-weight:750;letter-spacing:2px;text-transform:uppercase;}
.cq-callout {background:#edf6f5;border-left:4px solid #087f8c;padding:13px 18px;margin:16px 0;color:#18324d;}
.cq-table {border-collapse:collapse;width:100%;font-size:14px;margin:14px 0 18px;}
.cq-table th {background:#18324d;color:white;text-align:left;padding:10px 12px;}
.cq-table td {padding:9px 12px;border-bottom:1px solid #dce4e8;vertical-align:top;text-align:left;white-space:normal;}
.cq-table tr:nth-child(even) {background:#f3f7f8;}
.cq-small {color:#627488;font-size:12px;line-height:1.5;}
.cq-flow {display:flex;gap:9px;align-items:stretch;margin:20px 0;flex-wrap:wrap;}
.cq-flow div {flex:1;background:#edf6f5;border-top:3px solid #087f8c;padding:13px;min-width:130px;color:#18324d;}
blockquote {border-left:3px solid #ccd9df;padding-left:15px;color:#576c7d;font-size:.9em;}
@media print {.jp-InputArea {display:none;} h2 {break-before:page;} table,figure {break-inside:avoid;}}
</style>
"""


def table(frame: pd.DataFrame, formats: dict | None = None) -> None:
    """Escape product text while retaining a compact, legible presentation table."""
    display(HTML(frame.to_html(index=False, border=0, classes="cq-table", escape=True,
                              formatters=formats or {})))


def show_figure(fig) -> None:
    fig.tight_layout()
    output = io.StringIO()
    fig.savefig(output, format="svg", bbox_inches="tight", metadata={"Date": None})
    display(SVG(output.getvalue()))
    plt.close(fig)


def fingerprint(ids) -> str:
    digest = hashlib.sha256()
    for value in ids:
        payload = str(value).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class Review:
    """Load a small, fixed evidence set and fail if identity/metric checks disagree."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.consumed: dict[str, str] = {}
        self.allowed = {
            f"{SPLIT}/{name}" for name in
            ("train.csv", "validation.csv", "assignments.csv", "protected_final_test.csv")
        }
        self.allowed.update(f"{folder}/assignments.csv" for folder in ALLOCATIONS.values())
        self.allowed.update({
            f"{CLEAN}/validation.json", f"{CLEAN}/label_changes.csv",
            f"{REMAINING}/brand_result.json",
            f"{EXPORT}/brand_lookup_probabilistic_v1/evaluation.json",
            f"{CATEGORY}/validation_comparison.csv",
            f"{TRANSFER}/validation_metrics.csv",
            f"{TRANSFER}/paired_differences.csv",
            "data/processed/minilm_representation_audit_20261015/audit.json",
            "docs/findings/00_data_profile/01_profile_results.md",
            "docs/findings/03_modeling/02_classical_segment_results_20261010.md",
        })
        for slug in [*TARGETS, "brand_lookup_probabilistic_v1"]:
            self.allowed.update(f"{EXPORT}/{slug}/{name}" for name in (
                "manifest.json", "predictions.csv", "validation_context.csv",
                "harness_compatibility.json", "label_mapping.json"))
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.left": False, "axes.spines.bottom": False,
            "axes.labelcolor": NAVY, "text.color": NAVY, "xtick.color": GREY,
            "ytick.color": NAVY, "figure.facecolor": "white", "axes.facecolor": "white",
            "figure.dpi": 125, "axes.titleweight": "bold", "axes.titlepad": 14,
        })
        self._load_and_verify()

    def read(self, relative: str) -> bytes:
        if relative not in self.allowed or Path(relative).name == "test.csv":
            raise PermissionError(f"Outside the presentation evidence allowlist: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError(relative)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing existing artifact: {relative}. Restore the approved private "
                "artifact; this notebook deliberately does not generate replacements.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if relative in self.consumed:
            assert self.consumed[relative] == digest, f"Artifact changed: {relative}"
        self.consumed[relative] = digest
        return payload

    def csv(self, relative: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read(relative)), keep_default_na=False, **kwargs)

    def json(self, relative: str):
        return json.loads(self.read(relative).decode("utf-8"))

    def _load_and_verify(self):
        # Metadata-only registry: protected labels and test.csv are never loaded or hashed.
        protected = self.csv(f"{SPLIT}/protected_final_test.csv", dtype=str,
                             usecols=["record_id", "group_id", "split"])
        assert len(protected) == 12160 and protected.record_id.is_unique
        assert protected.split.eq("test").all()
        self.protected_ids, self.protected_groups = set(protected.record_id), set(protected.group_id)
        columns = ["record_id", "source_row", "group_id", "split", "ProductName",
                   "ProductBrand", "ProductCategory", "ProductDescription", "ProductContents",
                   "Segment", "Mnfr", "TargetAgeGroup", "Brand", "Sub-Segment"]
        frames = [self.csv(f"{SPLIT}/{name}.csv", dtype=str, usecols=columns)
                  for name in ("train", "validation")]
        assert [len(f) for f in frames] == [59111, 12667]
        for role, frame in zip(("train", "validation"), frames):
            assert frame.split.eq(role).all()
            assert not (set(frame.record_id) & self.protected_ids)
            assert not (set(frame.group_id) & self.protected_groups)
        assert not (set(frames[0].group_id) & set(frames[1].group_id))
        self.development = pd.concat(frames, ignore_index=True)
        assert self.development.record_id.is_unique
        self.dev_index = self.development.set_index("record_id")
        assignments = self.csv(f"{SPLIT}/assignments.csv", dtype=str)
        assert assignments.record_id.is_unique and len(assignments) == 83938
        assert assignments.groupby("group_id").split.nunique().max() == 1
        assigned_dev = assignments.set_index("record_id").loc[self.dev_index.index]
        assert assigned_dev.split.tolist() == self.dev_index.split.tolist()
        assert assigned_dev.group_id.tolist() == self.dev_index.group_id.tolist()
        self.assignments = {"V4": assignments}
        for target, folder in ALLOCATIONS.items():
            allocation = self.csv(f"{folder}/assignments.csv", dtype=str)
            assert allocation.record_id.is_unique and len(allocation) == 83938
            assert set(allocation.record_id) == set(assignments.record_id)
            assert allocation.groupby("group_id").split.nunique().max() == 1
            original = assignments.set_index("record_id").loc[allocation.record_id]
            assert original.group_id.tolist() == allocation.group_id.tolist()
            assert set(allocation.loc[allocation.split.eq("test"), "record_id"]) == self.protected_ids
            assert set(allocation.loc[allocation.split.ne("test"), "record_id"]) == set(self.dev_index.index)
            self.assignments[target] = allocation
        self.cleaning_validation = self.json(f"{CLEAN}/validation.json")
        self.changes = self.csv(f"{CLEAN}/label_changes.csv", dtype=str)
        assert len(self.changes) == 7
        assert set(self.changes.source_row).issubset(set(self.development.source_row))
        self.equality = self.json("data/processed/minilm_representation_audit_20261015/audit.json")
        assert self.equality["records"] == 83938 and self.equality["crossing_pairs"] == 0
        assert self.equality["final_test_target_columns_read"] == []
        self.predictions, self.manifests, self.per_class = {}, {}, {}
        self.score_rows = []
        for slug, (target, rows, classes) in TARGETS.items():
            self._verify_export(slug, target, rows, classes)
        self._verify_export("brand_lookup_probabilistic_v1", "Brand", 12619, 143)
        self.scores = pd.DataFrame(self.score_rows)
        self.brand_result = self.json(f"{REMAINING}/brand_result.json")
        self.lookup_evaluation = self.json(f"{EXPORT}/brand_lookup_probabilistic_v1/evaluation.json")
        original = self.brand_result["lookup_baseline"]["full_validation"]
        reproduced = self.lookup_evaluation["deterministic_reproduced"]
        for key in ("accuracy", "macro_f1", "macro_f1_validation_supported_classes"):
            assert np.isclose(original[key], reproduced[key], atol=1e-12, rtol=0)
        self.category = self.csv(f"{CATEGORY}/validation_comparison.csv")
        self.transfer = self.csv(f"{TRANSFER}/validation_metrics.csv")
        self.transfer_intervals = self.csv(f"{TRANSFER}/paired_differences.csv")
        gap = self.transfer_intervals.loc[
            self.transfer_intervals.metric.eq("accuracy")
            & self.transfer_intervals.cohort.eq("retailer_holdout")].iloc[0]
        assert round(-gap.left_minus_right * 100, 2) == 7.37
        assert round(-gap.upper * 100, 2) == 5.57
        assert round(-gap.lower * 100, 2) == 9.44
        self.profile_text = self.read("docs/findings/00_data_profile/01_profile_results.md").decode("utf-8")
        self.classical_text = self.read("docs/findings/03_modeling/02_classical_segment_results_20261010.md").decode("utf-8")

    def _verify_export(self, slug, target, rows, classes):
        folder = f"{EXPORT}/{slug}"
        manifest = self.json(f"{folder}/manifest.json")
        assert manifest["target"] == target and manifest["partition"] == "validation"
        assert manifest["validation_rows"] == rows and len(manifest["label_names"]) == classes
        for name in ("predictions.csv", "validation_context.csv", "label_mapping.json", "harness_compatibility.json"):
            actual = hashlib.sha256(self.read(f"{folder}/{name}")).hexdigest()
            assert actual == manifest["files_sha256"][name], f"Saved seal mismatch: {folder}/{name}"
        predictions = self.csv(f"{folder}/predictions.csv", dtype={
            "record_id": str, "true_label": str, "predicted_label": str})
        context = self.csv(f"{folder}/validation_context.csv", dtype=str)
        assert len(predictions) == rows and predictions.record_id.is_unique
        assert context.record_id.is_unique and context.split.eq("validation").all()
        assert context.record_id.tolist() == predictions.record_id.tolist()
        assert context.true_label.tolist() == predictions.true_label.tolist()
        assert not (set(context.record_id) & self.protected_ids)
        assert not (set(context.group_id) & self.protected_groups)
        allocation = self.assignments.get(target, self.assignments["V4"])
        valid_ids = allocation.loc[allocation.split.eq("validation"), "record_id"]
        truth = self.dev_index.loc[valid_ids, target]
        eligible = truth[~truth.str.strip().str.casefold().isin(["", "null", "none", "nan"])]
        assert set(predictions.record_id) == set(eligible.index)
        expected = self.dev_index.loc[predictions.record_id]
        assert expected[target].tolist() == predictions.true_label.tolist()
        assert expected.group_id.tolist() == context.group_id.tolist()
        assert fingerprint(sorted(predictions.record_id)) == manifest["validation_record_ids_sha256"]
        labels = manifest["label_names"]
        mapping = self.json(f"{folder}/label_mapping.json")
        probability_columns = [f"probability_{i}" for i in range(classes)]
        assert mapping == dict(zip(probability_columns, labels))
        assert predictions.columns.tolist() == ["record_id", "true_label", "predicted_label", "confidence", *probability_columns]
        probabilities = predictions[probability_columns].to_numpy(float)
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
        assert np.array_equal(np.asarray(labels)[probabilities.argmax(axis=1)], predictions.predicted_label)
        assert np.allclose(probabilities.max(axis=1), predictions.confidence, atol=1e-6, rtol=0)
        accuracy = accuracy_score(predictions.true_label, predictions.predicted_label)
        macro = f1_score(predictions.true_label, predictions.predicted_label, labels=labels, average="macro", zero_division=0)
        observed = f1_score(predictions.true_label, predictions.predicted_label, average="macro", zero_division=0)
        harness = self.json(f"{folder}/harness_compatibility.json")
        harness = harness.get("loader_and_scoring", harness)
        assert harness["status"] == "compatible" and harness["rows_consumed"] == rows
        assert harness["protected_test_records_scored"] == 0
        if slug in TARGETS:
            expected_metrics = harness["metrics"]
            for key, value in [("accuracy", accuracy), ("macro_f1_full_vocabulary", macro),
                               ("macro_f1_observed_labels", observed)]:
                assert np.isclose(value, expected_metrics[key], atol=1e-10, rtol=0), (target, key)
            train_ids = allocation.loc[allocation.split.eq("train"), "record_id"]
            train_labels = self.dev_index.loc[train_ids, target]
            train_eligible = ~train_labels.str.strip().str.casefold().isin(["", "null", "none", "nan"])
            train_count = train_eligible.sum()
            assert set(train_labels[train_eligible]) == set(labels)
            self.score_rows.append({"Target": target, "Labeled train": int(train_count),
                "Validation n": rows, "Classes train / val": f"{classes} / {predictions.true_label.nunique()}",
                "Accuracy": accuracy, "Macro-F1 (full)": macro, "Macro-F1 (observed)": observed})
        else:
            assert np.isclose(macro, manifest["evaluation"]["macro_f1"], atol=1e-10, rtol=0)
            assert predictions.record_id.tolist() == self.predictions["brand"].record_id.tolist()
            assert predictions.true_label.tolist() == self.predictions["brand"].true_label.tolist()
        self.manifests[slug] = manifest
        self.predictions[slug] = predictions
        report = classification_report(predictions.true_label, predictions.predicted_label,
                                       labels=labels, output_dict=True, zero_division=0)
        self.per_class[slug] = pd.DataFrame([
            {"Label": label, **report[label]} for label in labels])

    def profile(self):
        section = self.profile_text.split("## Missing Values")[1].split("### Target")[0]
        findings = dict(re.findall(r"- `([^`]+)`: ([\d,]+) missing", section))
        selected = ["Platform", "Upc", "ProductDescription", "ProductContents", "Sub-Segment", "TargetAgeGroup"]
        values = [int(findings[label].replace(",", "")) / 84552 * 100 for label in selected]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.35), gridspec_kw={"width_ratios": [1.2, 1]})
        axes[0].barh(selected[::-1], values[::-1], color=TEAL, height=.6)
        axes[0].set(xlim=(0, 110), xlabel="Missing in original training export (%)", title="Missing fields are not missing products")
        for i, value in enumerate(values[::-1]): axes[0].text(value + 1, i, f"{value:.1f}%", va="center", fontsize=10)
        validation = self.development[self.development.split.eq("validation")]
        counts = validation.Segment[~validation.Segment.isin(["", "null"])].value_counts()
        assert counts.sum() == 11952 and counts.iloc[0] == 7380
        short = ["Vitamins / minerals / supplements" if s.startswith("Vitamins,") else s for s in counts.index]
        axes[1].barh(short[::-1], counts.values[::-1], color=NAVY, height=.6)
        axes[1].set(xlabel="Labeled Segment validation records", title="Class imbalance survives cleaning")
        axes[1].text(.98, .04, "7,380 / 11,952 = 61.75% in the largest class", transform=axes[1].transAxes,
                     ha="right", fontsize=9, color=GREY)
        show_figure(fig)

    def cleaning(self):
        rows = []
        for role, title in [("training", "Labeled source"), ("target", "Prediction-input source")]:
            item = self.cleaning_validation[role]
            rows.append({"Source": title, "Original rows": item["verified_rows"],
                         "Candidate": item["counts"]["candidate"], "Quarantine": item["counts"]["quarantine"],
                         "Review hold": item["counts"]["review_hold"]})
        table(pd.DataFrame(rows))
        examples = self.changes[self.changes.source_row.isin(["21582", "37965"])].merge(
            self.development[["source_row", "ProductName", "Sub-Segment"]], on="source_row", validate="one_to_one")
        assert len(examples) == 2 and examples["new_value"].tolist() == examples["Sub-Segment"].tolist()
        table(examples[["source_row", "ProductName", "old_value", "new_value"]].rename(columns={
            "source_row": "Source row (1-based)", "ProductName": "Real development product",
            "old_value": "Before", "new_value": "After"}))

    def split_table(self):
        table(pd.DataFrame([
            {"Partition": "Train", "All rows": 59111, "Share": "70.42%", "Use now": "Fit models / training-only mappings"},
            {"Partition": "Validation", "All rows": 12667, "Share": "15.09%", "Use now": "Compare development candidates"},
            {"Partition": "Protected test", "All rows": 12160, "Share": "14.49%", "Use now": "Reserved; IDs/groups only in this notebook"},
        ]))

    def baseline(self):
        # Classical values are from the committed verified report, not a new fit.
        expected_lines = [
            "| Training-selected majority class | 61.75% | 9.54% | 4,572 |",
            "| Four agreed product fields | 89.57% | 83.90% | 1,246 |",
            "| Four fields + concatenated ProductCategory | 94.42% | 91.00% | 667 |",
        ]
        assert all(line in self.classical_text for line in expected_lines)
        data = pd.DataFrame({
            "Candidate": ["Majority", "TF-IDF · 4 fields", "TF-IDF · + category", "MiniLM · 4 fields", "MiniLM · + category"],
            "Accuracy": [1 - 4572/11952, 1 - 1246/11952, 1 - 667/11952,
                         self.category.iloc[0].accuracy, self.category.iloc[1].accuracy],
            "Macro-F1": [.0954, .8390, .9100, self.category.iloc[0].macro_f1, self.category.iloc[1].macro_f1],
        })
        fig, ax = plt.subplots(figsize=(11.6, 3.4))
        y = np.arange(len(data))
        ax.barh(y-.16, data.Accuracy*100, height=.29, color=NAVY, label="Accuracy")
        ax.barh(y+.16, data["Macro-F1"]*100, height=.29, color=TEAL, label="Macro-F1")
        ax.set(yticks=y, yticklabels=data.Candidate, xlim=(0, 109), xlabel="Validation score (%) · identical 11,952 records")
        ax.invert_yaxis()
        for i, row in data.iterrows():
            ax.text(row.Accuracy*100+.6, i-.16, f"{row.Accuracy:.2%}", va="center", fontsize=9)
            ax.text(row["Macro-F1"]*100+.6, i+.16, f"{row['Macro-F1']:.2%}", va="center", fontsize=9)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, ncol=2)
        show_figure(fig)

    def results(self):
        table(self.scores, {key: lambda value: f"{value:.2%}" for key in
                           ("Accuracy", "Macro-F1 (full)", "Macro-F1 (observed)")})
        fig, ax = plt.subplots(figsize=(11.6, 2.8))
        y = np.arange(len(self.scores))
        ax.barh(y-.16, self.scores.Accuracy*100, height=.28, color=NAVY, label="Accuracy")
        ax.barh(y+.16, self.scores["Macro-F1 (full)"]*100, height=.28, color=TEAL, label="Macro-F1 · full vocabulary")
        ax.set(yticks=y, yticklabels=self.scores.Target, xlim=(0, 105), xlabel="Validation score (%) · target-specific populations")
        ax.invert_yaxis()
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, ncol=2)
        show_figure(fig)

    def weak_classes(self):
        focus = {
            "mnfr": ["J&J"], "targetagegroup": ["Infant", "Children"],
            "sub_segment": ["External Upper Respiratory", "Other Allergy", "Temperature Packs"],
            "brand": ["Astepro"],
        }
        rows = []
        for slug, labels in focus.items():
            frame = self.per_class[slug].set_index("Label")
            for label in labels:
                item = frame.loc[label]
                rows.append({"Target": TARGETS[slug][0], "Class": label, "Validation n": int(item.support),
                             "Precision": item.precision, "Recall": item.recall, "F1": item["f1-score"]})
        table(pd.DataFrame(rows), {key: lambda v: f"{v:.2%}" for key in ("Precision", "Recall", "F1")})

    def brand(self):
        deterministic = self.brand_result["lookup_baseline"]["full_validation"]
        probabilistic = self.lookup_evaluation["probabilistic"]
        neural = self.scores[self.scores.Target.eq("Brand")].iloc[0]
        frame = pd.DataFrame([
            {"Method": "MiniLM", "Accuracy": neural.Accuracy, "Macro-F1 · all 143": neural["Macro-F1 (full)"],
             "Macro-F1 · supported 138": neural["Macro-F1 (observed)"]},
            {"Method": "Deterministic lookup + fallback", "Accuracy": deterministic["accuracy"],
             "Macro-F1 · all 143": deterministic["macro_f1"],
             "Macro-F1 · supported 138": deterministic["macro_f1_validation_supported_classes"]},
            {"Method": "Probabilistic lookup (fixed α=1)", "Accuracy": probabilistic["accuracy"],
             "Macro-F1 · all 143": probabilistic["macro_f1"],
             "Macro-F1 · supported 138": probabilistic["macro_f1_validation_supported_classes"]},
        ])
        table(frame, {c: lambda v: f"{v:.3%}" for c in frame.columns if c != "Method"})
        cohorts = self.lookup_evaluation["cohorts"]
        labels = {"seen_feature_values": "Seen keys", "unseen_nonmissing_values": "Unseen keys",
                  "missing_values": "Missing ProductBrand", "all_fallback_records": "All fallback (subtotal)"}
        table(pd.DataFrame([{"Lookup cohort": labels[k], "Records": cohorts[k]["records"],
            "Deterministic accuracy": cohorts[k]["deterministic"]["accuracy"],
            "Probabilistic accuracy": cohorts[k]["probabilistic"]["accuracy"]} for k in labels]),
            {c: lambda v: f"{v:.2%}" for c in ("Deterministic accuracy", "Probabilistic accuracy")})

    def transfer_chart(self):
        focus = self.transfer[self.transfer.cohort.eq("retailer_holdout")].set_index("training_regime")
        table(pd.DataFrame([
            {"Training population": label, "Same non-Amazon validation n": int(focus.loc[key, "records"]),
             "Accuracy": focus.loc[key, "accuracy"], "Macro-F1": focus.loc[key, "macro_f1"]}
            for key, label in [("mixed_retailer_reference", "Mixed retailers (reference)"), ("amazon_only", "Amazon only")]
        ]), {"Accuracy": lambda v: f"{v:.2%}", "Macro-F1": lambda v: f"{v:.2%}"})
        cohorts = ["amazon_control", "retailer_holdout"]
        fig, ax = plt.subplots(figsize=(11.6, 2.9))
        x = np.arange(2)
        for offset, regime, label, color in [(-.17, "mixed_retailer_reference", "Mixed-retailer training", NAVY),
                                            (.17, "amazon_only", "Amazon-only training", TEAL)]:
            scores = [float(self.transfer.loc[self.transfer.training_regime.eq(regime) & self.transfer.cohort.eq(c), "accuracy"].iloc[0])*100 for c in cohorts]
            bars = ax.bar(x+offset, scores, width=.30, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in scores], padding=4, fontsize=10)
        ax.set(xticks=x, xticklabels=["Amazon control · n=9,441", "Other retailers · n=2,511"], ylim=(0, 109), ylabel="Accuracy (%)")
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, ncol=2)
        show_figure(fig)

    def examples(self):
        # Fixed development examples, chosen for discussion rather than sampled randomly.
        selections = [
            ("segment", "06d2ca12999a801a09e50c57c1d86d38aeb17aa4a46e419dcbee516450bc63b0"),
            ("sub_segment", "11a8bd38c1a4503ab677d6c0d9f7e74e218d2de8916c6a68ba53a3be2f502b07"),
            ("brand", "0b29472b272d97f20a4a6b7946f4da188695911a3b1d5a685801dc006bbb4f9f"),
        ]
        rows, self.example_ids = [], []
        for slug, record_id in selections:
            assert record_id not in self.protected_ids
            prediction = self.predictions[slug].set_index("record_id").loc[record_id]
            product = self.dev_index.loc[record_id]
            title = product.ProductName
            rows.append({"Target / ID prefix": f"{TARGETS[slug][0]} · {record_id[:10]}",
                         "Real validation product (title excerpt)": title[:95] + ("…" if len(title)>95 else ""),
                         "Supplied label": prediction.true_label, "MiniLM": prediction.predicted_label,
                         "Uncalibrated confidence": prediction.confidence})
            self.example_ids.append({"target": TARGETS[slug][0], "record_id": record_id})
        table(pd.DataFrame(rows), {"Uncalibrated confidence": lambda v: f"{v:.1%}"})
        brand_id = selections[-1][1]
        lookup = self.predictions["brand_lookup_probabilistic_v1"].set_index("record_id").loc[brand_id]
        assert lookup.predicted_label == "Astepro"
        display(HTML('<p class="cq-small">For the Astepro record, the saved probabilistic lookup predicts Astepro correctly. '
                     'These deliberately selected errors illustrate failure modes; they do not estimate error prevalence.</p>'))

    def verify_unchanged(self) -> dict:
        for relative, digest in list(self.consumed.items()):
            assert hashlib.sha256(self.read(relative)).hexdigest() == digest
        return {
            "status": "passed", "consumed_files_unchanged": len(self.consumed),
            "minilm_target_rows_recomputed": sum(len(self.predictions[s]) for s in TARGETS),
            "probabilistic_lookup_rows_recomputed": len(self.predictions["brand_lookup_probabilistic_v1"]),
            "metrics_match_saved_results": True, "validation_membership_and_labels_match_development": True,
            "probability_class_order_and_sums_verified": True, "protected_id_or_group_overlap": 0,
            "protected_test_labels_read": False, "test_csv_opened_or_hashed": False,
            "model_training_or_inference_performed": False,
            "consumed_sha256": dict(sorted(self.consumed.items())),
        }
