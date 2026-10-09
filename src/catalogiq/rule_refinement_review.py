"""Freeze blinded reviews before evaluating a rule-only refinement.

The reserved sample is a lexical challenge set, not a probability sample. All
reserved pairs receive two reviews; unresolved disagreements remain uncertain.
Product identity is retained separately from the leakage judgment used to score.
This module never proposes matches, constructs groups or allocates splits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .balanced_pair_sample import EVIDENCE_FIELDS, PAIR_COLUMNS, REVIEW_FIELDS
from .balanced_review_evaluation import adjudicate_secondary_reviews, validate_balanced_reviews
from .features import sha256
from .grouping_regression import evaluate_review_groups
from .independent_review_reservation import verify_reservation
from .splitting import check_assignments

CAUTION = ('Independent, purposively retrieved challenge pairs; qualitative model-assisted '
           'text review, not externally verified identity or population-wide accuracy. '
           'Provisional retrieval strata are not gold classes; uncertain cases stay separate.')
METHODS = ('rule_v2', 'rule_v3', 'tfidf_v2')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


def _verify_method(run_dir):
    from .rule_refinement_experiment import verify_rule_refinement_freeze
    return verify_rule_refinement_freeze(Path(run_dir))


def _verify_hashes(directory, hashes):
    for name, digest in hashes.items():
        path = Path(directory) / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'sealed review artifact changed: {name}')


def prepare_review(reservation_dir, run_dir, review_dir):
    """Release neutral evidence only after the new grouping version is frozen."""
    reservation_dir, run_dir, review_dir = map(Path, (reservation_dir, run_dir, review_dir))
    if review_dir.exists():
        raise FileExistsError('refusing to overwrite an independent review bundle')
    verify_reservation(reservation_dir)
    _verify_method(run_dir)
    rows = _json(reservation_dir / 'blind_pairs.json')
    allowed = {'pair_id', *PAIR_COLUMNS,
               *(f'{side}_{field}' for side in ('left', 'right') for field in EVIDENCE_FIELDS)}
    if not rows or any(set(row) != allowed for row in rows):
        raise ValueError('blind evidence must contain only neutral identifiers and product features')
    registry = pd.read_csv(reservation_dir / 'reserved_pairs.csv', dtype=str, keep_default_na=False)
    keys = pd.DataFrame(rows)[['pair_id', *PAIR_COLUMNS]].sort_values('pair_id').reset_index(drop=True)
    expected = registry[['pair_id', *PAIR_COLUMNS]].sort_values('pair_id').reset_index(drop=True)
    if not keys.equals(expected):
        raise ValueError('blind evidence differs from reserved registry')
    review_dir.mkdir(parents=True)
    # Register complete secondary coverage before either review starts. Chunks
    # are by neutral registry order, never confidence or method predictions.
    for prefix, chunks in (('primary', 3), ('secondary', 2)):
        for index in range(chunks):
            start, stop = len(rows) * index // chunks, len(rows) * (index + 1) // chunks
            _write_json(review_dir / f'{prefix}_blind_{index + 1}.json', rows[start:stop])
    protocol = {
        'status': 'new_method_frozen_before_any_reserved_product_review',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method_run': str(run_dir.resolve()),
        'method_freeze_sha256': sha256(run_dir / 'rule_refinement_freeze.json'),
        'reservation_dir': str(reservation_dir.resolve()),
        'reservation_summary_sha256': sha256(reservation_dir / 'summary.json'),
        'reserved_pairs': len(rows), 'primary_coverage': 'all reserved pairs',
        'secondary_coverage': 'all reserved pairs; registered before primary judgments',
        'identity_policy': 'Same sellable item including formulation, physical form, flavor, strength and packaging; missing information may make identity uncertain.',
        'leakage_policy': 'Keep together for distinctive named-line/model lineage or substantive near-copied product templates that shortcut Segment prediction; generic category vocabulary and supplier marketing alone are insufficient. Different formulations can still leak.',
        'review_features': EVIDENCE_FIELDS,
        'hidden': ['targets', 'methods', 'group membership', 'split assignments',
                   'matching scores', 'retrieval strata', 'prior judgments'],
        'adjudication': 'Different leakage decisions stay uncertain; differing confidence retains the more cautious confidence. Never force category balance.',
        'scoring_policy': 'Preserve and score primary-only judgments separately from adjudicated judgments. Secondary review is a sensitivity check, not additional independent examples.',
        'review_gate': 'Freeze both primary and secondary judgments and adjudication before scoring any reserved pair.',
        'no_retuning': 'No grouping implementation/configuration changes after review; subsequent changes require new independent examples.',
        'interpretation': CAUTION,
        'blind_sha256': {path.name: sha256(path) for path in sorted(review_dir.iterdir())},
    }
    _write_json(review_dir / 'protocol.json', protocol)
    return {'reserved_pairs': len(rows), 'frozen_before_review': True, 'method_outcomes_released': False}


def _review_annotations(paths, evidence):
    rows = []
    for path in paths:
        rows.extend(_json(path))
    reviews = pd.DataFrame(rows)
    required = {'pair_id', *PAIR_COLUMNS, *REVIEW_FIELDS}
    if not required <= set(reviews):
        raise ValueError('review annotations lack separate identity/leakage judgments')
    if set(reviews) - required - {'left_ProductName', 'right_ProductName'}:
        raise ValueError('annotation contains unapproved outcomes or labels')
    if reviews.pair_id.duplicated().any() or set(reviews.pair_id) != set(evidence.pair_id):
        raise ValueError('every reserved pair must be reviewed exactly once in each review pass')
    reviews = reviews.set_index('pair_id').loc[evidence.pair_id].reset_index()
    for column in PAIR_COLUMNS:
        if list(reviews[column]) != list(evidence[column]):
            raise ValueError('review endpoint differs from reserved evidence')
    for column in ('left_ProductName', 'right_ProductName'):
        if column in reviews and list(reviews[column]) != list(evidence[column]):
            raise ValueError('review names differ from reserved evidence')
        reviews[column] = list(evidence[column])
    validate_balanced_reviews(reviews)
    return reviews


def freeze_reviews(review_dir, primary_paths, secondary_paths):
    """Keep all pairs and all ambiguity; no outcome-based selection occurs."""
    review_dir = Path(review_dir)
    freeze_path = review_dir / 'judgment_freeze.json'
    if freeze_path.exists():
        raise FileExistsError('refusing to replace frozen independent judgments')
    protocol = _json(review_dir / 'protocol.json')
    run_dir = Path(protocol['method_run'])
    _verify_method(run_dir)
    if sha256(run_dir / 'rule_refinement_freeze.json') != protocol['method_freeze_sha256']:
        raise ValueError('new method changed after release to reviewers')
    verify_reservation(protocol['reservation_dir'])
    _verify_hashes(review_dir, protocol['blind_sha256'])
    evidence = pd.DataFrame([row for name in sorted(protocol['blind_sha256'])
                            if name.startswith('primary_') for row in _json(review_dir / name)])
    primary_paths, secondary_paths = list(map(Path, primary_paths)), list(map(Path, secondary_paths))
    if any(not path.resolve().is_relative_to(review_dir.resolve()) for path in primary_paths + secondary_paths):
        raise ValueError('retain source annotation files within the review bundle')
    primary = _review_annotations(primary_paths, evidence)
    secondary = _review_annotations(secondary_paths, evidence)
    reviewed, agreement = adjudicate_secondary_reviews(primary, secondary)
    validate_balanced_reviews(reviewed)
    primary.to_csv(review_dir / 'primary_reviewed_pairs.csv', index=False)
    secondary.to_csv(review_dir / 'secondary_reviewed_pairs.csv', index=False)
    reviewed.to_csv(review_dir / 'annotated_reserved_pairs.csv', index=False)
    agreement.to_csv(review_dir / 'review_agreement.csv', index=False)
    names = ['protocol.json', *protocol['blind_sha256'], 'primary_reviewed_pairs.csv',
             'secondary_reviewed_pairs.csv', 'annotated_reserved_pairs.csv', 'review_agreement.csv']
    paths = [review_dir / name for name in names] + primary_paths + secondary_paths
    freeze = {'status': 'independent_judgments_frozen_before_method_scoring',
              'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'pairs': len(reviewed), 'category_counts': reviewed.review_category.value_counts().to_dict(),
              'leakage_counts': reviewed.leakage_decision.value_counts().to_dict(),
              'method_freeze_sha256': protocol['method_freeze_sha256'], 'interpretation': CAUTION,
              'review_code_sha256': {str(Path(__file__).resolve()): sha256(Path(__file__)),
                  str(Path(__file__).with_name('balanced_review_evaluation.py').resolve()):
                  sha256(Path(__file__).with_name('balanced_review_evaluation.py'))},
              'sha256': {str(path.resolve().relative_to(review_dir.resolve())): sha256(path) for path in paths}}
    _write_json(freeze_path, freeze)
    return freeze


def score_reviews(reviews, assignments):
    """Generic comparison of three fixed partitions; identity never scores errors."""
    validate_balanced_reviews(reviews)
    if set(assignments) != set(METHODS):
        raise ValueError('comparison requires original rule, refined rule and frozen TF-IDF')
    rows, metrics = [], []
    universe = None
    for method in METHODS:
        assignment = assignments[method]
        check_assignments(assignment, assignment.record_id)
        ids = set(assignment.record_id)
        if universe is not None and ids != universe:
            raise ValueError('methods must cover identical records')
        universe = ids
        scored = evaluate_review_groups(reviews, assignment)
        indexed = assignment.set_index('record_id')
        scored['method'] = method
        scored['crosses_split'] = scored.left_record_id.map(indexed.split).ne(scored.right_record_id.map(indexed.split))
        expected = scored.leakage_decision.map({'keep_together': True, 'keep_separate': False}).astype('boolean')
        scored['actual_cross_split_leakage'] = (expected & scored.crosses_split).where(expected.notna())
        sizes = indexed.groupby('group_id').size()
        for side in ('left', 'right'):
            scored[f'{side}_group_id'] = scored[f'{side}_record_id'].map(indexed.group_id)
            scored[f'{side}_group_size'] = scored[f'{side}_group_id'].map(sizes)
        rows.append(scored)
        for category in ('decisive_core', 'related', 'safe_to_separate', 'borderline'):
            subset = scored.loc[scored.review_category.ne('borderline') if category == 'decisive_core'
                                else scored.review_category.eq(category)]
            positive = subset.leakage_decision.eq('keep_together')
            negative = subset.leakage_decision.eq('keep_separate')
            unknown = subset.leakage_decision.eq('uncertain')
            metrics.append({'method': method, 'review_category': category, 'pairs': len(subset),
                'keep_together_pairs': int(positive.sum()), 'keep_separate_pairs': int(negative.sum()),
                'uncertain_pairs': int(unknown.sum()),
                'false_negative_pairs': int(subset.false_negative.sum()),
                'false_positive_pairs': int(subset.false_positive.sum()),
                'actual_cross_split_leakage_pairs': int(subset.actual_cross_split_leakage.sum()),
                'uncertain_pairs_grouped': int(subset.loc[unknown, 'grouped'].sum()),
                'uncertain_pairs_separated': int((~subset.loc[unknown, 'grouped']).sum()),
                'interpretation': CAUTION})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def compare_reserved_review(review_dir, baseline_dir, output_dir):
    review_dir, baseline_dir, output_dir = map(Path, (review_dir, baseline_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite reserved-review comparison')
    freeze = _json(review_dir / 'judgment_freeze.json')
    _verify_hashes(review_dir, freeze['sha256'])
    for path, digest in freeze['review_code_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('review/scoring policy changed after judgments were frozen')
    protocol = _json(review_dir / 'protocol.json')
    run_dir = Path(protocol['method_run'])
    _verify_method(run_dir)
    if sha256(run_dir / 'rule_refinement_freeze.json') != freeze['method_freeze_sha256']:
        raise ValueError('method freeze changed after review')
    verify_reservation(protocol['reservation_dir'])
    if Path(_json(run_dir / 'rule_refinement_freeze.json')['baseline_dir']).resolve() != baseline_dir.resolve():
        raise ValueError('review comparison baseline differs from frozen experiment')
    reviews = pd.read_csv(review_dir / 'annotated_reserved_pairs.csv', dtype=str, keep_default_na=False)
    assignments = {method: pd.read_csv(path, usecols=['record_id', 'group_id', 'split'], dtype=str, keep_default_na=False)
                   for method, path in {'rule_v2': baseline_dir / 'rule_assignments.csv',
                       'rule_v3': run_dir / 'rule_assignments.csv', 'tfidf_v2': baseline_dir / 'tfidf_assignments.csv'}.items()}
    outcomes, metrics = score_reviews(reviews, assignments)
    primary = pd.read_csv(review_dir / 'primary_reviewed_pairs.csv', dtype=str, keep_default_na=False)
    primary_outcomes, primary_metrics = score_reviews(primary, assignments)
    output_dir.mkdir(parents=True)
    outcomes.to_csv(output_dir / 'reserved_pair_outcomes.csv', index=False)
    metrics.to_csv(output_dir / 'reserved_metrics.csv', index=False)
    primary_outcomes.to_csv(output_dir / 'primary_pair_outcomes.csv', index=False)
    primary_metrics.to_csv(output_dir / 'primary_metrics.csv', index=False)
    errors = outcomes.loc[outcomes.reviewed_error.fillna(False),
        ['method', 'pair_id', 'review_category', 'left_ProductName', 'right_ProductName',
         'false_negative', 'false_positive', 'actual_cross_split_leakage',
         'left_group_size', 'right_group_size', 'leakage_rationale']]
    errors.to_csv(output_dir / 'product_error_examples.csv', index=False)
    primary_errors = primary_outcomes.loc[primary_outcomes.reviewed_error.fillna(False),
        ['method', 'pair_id', 'review_category', 'left_ProductName', 'right_ProductName',
         'false_negative', 'false_positive', 'actual_cross_split_leakage',
         'left_group_size', 'right_group_size', 'leakage_rationale']]
    primary_errors.to_csv(output_dir / 'primary_product_error_examples.csv', index=False)
    _write_json(output_dir / 'summary.json', {'method_frozen_before_review': True,
        'judgments_frozen_before_scoring': True, 'identity_did_not_determine_scoring': True,
        'pairs': len(reviews), 'review_categories': reviews.review_category.value_counts().to_dict(),
        'primary_review_categories': primary.review_category.value_counts().to_dict(),
        'reviewer_context': protocol.get('reviewer_context', {}),
        'primary_only_scored_separately': True,
        'method_freeze_sha256': freeze['method_freeze_sha256'],
        'judgment_freeze_sha256': sha256(review_dir / 'judgment_freeze.json'),
        'interpretation': CAUTION,
        'output_sha256': {path.name: sha256(path) for path in sorted(output_dir.iterdir())}})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--reservation-dir', type=Path, required=True)
    prepare.add_argument('--run-dir', type=Path, required=True)
    prepare.add_argument('--review-dir', type=Path, required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--review-dir', type=Path, required=True)
    freeze.add_argument('--primary', type=Path, action='append', required=True)
    freeze.add_argument('--secondary', type=Path, action='append', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--review-dir', type=Path, required=True)
    compare.add_argument('--baseline-dir', type=Path, required=True)
    compare.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        print(json.dumps(prepare_review(args.reservation_dir, args.run_dir, args.review_dir)))
    elif args.command == 'freeze':
        summary = freeze_reviews(args.review_dir, args.primary, args.secondary)
        print(json.dumps({key: summary[key] for key in ('pairs', 'category_counts', 'leakage_counts')}))
    else:
        metrics = compare_reserved_review(args.review_dir, args.baseline_dir, args.output_dir)
        print(metrics.loc[metrics.review_category.eq('decisive_core')].drop(columns='interpretation').to_string(index=False))
        print(CAUTION)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
