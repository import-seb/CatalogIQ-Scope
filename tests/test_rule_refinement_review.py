import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from catalogiq.balanced_pair_sample import EVIDENCE_FIELDS
from catalogiq.features import sha256
from catalogiq.rule_refinement_review import (
    _review_annotations, freeze_reviews, prepare_review, score_reviews,
)


def evidence(pair_id='F001', left='a', right='b'):
    row = {'pair_id': pair_id, 'left_record_id': left, 'right_record_id': right}
    for side in ('left', 'right'):
        for field in EVIDENCE_FIELDS:
            row[f'{side}_{field}'] = f'Cedar RestWave {side}' if field == 'ProductName' else ''
    return row


def judgment(row, decision='keep_together', category='related'):
    return {**{key: row[key] for key in ('pair_id', 'left_record_id', 'right_record_id')},
        'product_identity': 'different_product', 'identity_detail': 'pack_size_variant',
        'identity_confidence': 'high', 'identity_rationale': 'Different package sizes.',
        'leakage_decision': decision, 'leakage_confidence': 'high' if category != 'borderline' else 'medium',
        'leakage_rationale': 'Distinctive named line repeats the product template.',
        'review_category': category, 'judgment_uses_Segment': False,
        'judgment_uses_method_outcomes': False}


def assignments(grouped=True):
    return pd.DataFrame({'record_id': ['a', 'b', 'c', 'd'],
        'group_id': ['ab', 'ab' if grouped else 'b', 'c', 'd'],
        'split': ['train', 'train' if grouped else 'test', 'validation', 'test']})


class IndependentRefinementReviewTests(unittest.TestCase):
    def test_identity_difference_does_not_override_leakage_truth(self):
        product = evidence()
        reviews = _review_annotations_from_rows([judgment(product)], [product])
        result, metrics = score_reviews(reviews, {'rule_v2': assignments(False),
            'rule_v3': assignments(), 'tfidf_v2': assignments(False)})
        self.assertEqual(list(result.false_negative), [True, False, True])
        self.assertEqual(list(metrics.loc[metrics.review_category.eq('decisive_core'), 'actual_cross_split_leakage_pairs']), [1, 0, 1])
        changed = reviews.copy()
        changed.product_identity = 'uncertain'
        changed.identity_detail = 'uncertain'
        other, _ = score_reviews(changed, {'rule_v2': assignments(False),
            'rule_v3': assignments(), 'tfidf_v2': assignments(False)})
        pd.testing.assert_series_equal(result.false_negative, other.false_negative)

    def test_unknown_pairs_stay_unscored_and_target_columns_are_ignored(self):
        product = evidence()
        reviews = _review_annotations_from_rows([judgment(product, 'uncertain', 'borderline')], [product])
        methods = {name: assignments() for name in ('rule_v2', 'rule_v3', 'tfidf_v2')}
        result, metrics = score_reviews(reviews, methods)
        self.assertTrue(result.reviewed_error.isna().all())
        self.assertEqual(int(metrics.false_positive_pairs.sum()), 0)
        for assignment in methods.values():
            assignment['Segment'] = ['mutated', 'opposite', 'x', 'y']
        altered, altered_metrics = score_reviews(reviews, methods)
        pd.testing.assert_frame_equal(result, altered)
        pd.testing.assert_frame_equal(metrics, altered_metrics)

    def test_invalid_assignments_or_changed_universe_rejected(self):
        row = evidence()
        reviews = _review_annotations_from_rows([judgment(row)], [row])
        methods = {name: assignments() for name in ('rule_v2', 'rule_v3', 'tfidf_v2')}
        methods['rule_v3'].loc[1, 'split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'crosses split'):
            score_reviews(reviews, methods)
        methods['rule_v3'] = assignments().iloc[:2]
        with self.assertRaisesRegex(ValueError, 'identical records'):
            score_reviews(reviews, methods)

    def test_complete_review_coverage_endpoint_names_and_hidden_fields_enforced(self):
        row = evidence()
        for change in ({'left_record_id': 'wrong'}, {'Segment': 'category'},
                       {'judgment_uses_method_outcomes': True}, {'left_ProductName': 'changed'}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    _review_annotations_from_rows([{**judgment(row), **change}], [row])
        with self.assertRaises(ValueError):
            _review_annotations_from_rows([judgment(row), judgment(row)], [row])

    @patch('catalogiq.rule_refinement_review.verify_reservation')
    @patch('catalogiq.rule_refinement_review._verify_method')
    def test_method_gate_precedes_blind_release_and_all_secondary_coverage_registered(self, method, reservation):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, run, review = root / 'source', root / 'run', root / 'review'
            source.mkdir()
            run.mkdir()
            rows = [evidence(f'F{i:03d}', f'a{i}', f'b{i}') for i in range(6)]
            (source / 'blind_pairs.json').write_text(json.dumps(rows))
            (source / 'summary.json').write_text('{}')
            (run / 'rule_refinement_freeze.json').write_text('{}')
            pd.DataFrame(rows)[['pair_id', 'left_record_id', 'right_record_id']].to_csv(source / 'reserved_pairs.csv', index=False)
            method.side_effect = ValueError('not frozen')
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                prepare_review(source, run, review)
            self.assertFalse(review.exists())
            method.side_effect = None
            summary = prepare_review(source, run, review)
            self.assertFalse(summary['method_outcomes_released'])
            protocol = json.loads((review / 'protocol.json').read_text())
            self.assertEqual(protocol['method_freeze_sha256'], sha256(run / 'rule_refinement_freeze.json'))
            for prefix in ('primary', 'secondary'):
                exposed = [r for name in protocol['blind_sha256'] if name.startswith(prefix)
                           for r in json.loads((review / name).read_text())]
                self.assertEqual(exposed, rows)

    @patch('catalogiq.rule_refinement_review.verify_reservation')
    @patch('catalogiq.rule_refinement_review._verify_method')
    def test_review_disagreement_remains_uncertain_and_seal_is_immutable(self, method, reservation):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, run, review = root / 'source', root / 'run', root / 'review'
            source.mkdir()
            run.mkdir()
            row = evidence()
            (source / 'blind_pairs.json').write_text(json.dumps([row]))
            (source / 'summary.json').write_text('{}')
            (run / 'rule_refinement_freeze.json').write_text('{}')
            pd.DataFrame([row])[['pair_id', 'left_record_id', 'right_record_id']].to_csv(source / 'reserved_pairs.csv', index=False)
            prepare_review(source, run, review)
            primary, secondary = review / 'primary.json', review / 'secondary.json'
            primary.write_text(json.dumps([judgment(row)]))
            secondary.write_text(json.dumps([judgment(row, 'keep_separate', 'safe_to_separate')]))
            sealed = freeze_reviews(review, [primary], [secondary])
            self.assertEqual(sealed['leakage_counts'], {'uncertain': 1})
            with self.assertRaises(FileExistsError):
                freeze_reviews(review, [primary], [secondary])


def _review_annotations_from_rows(reviews, products):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'annotations.json'
        path.write_text(json.dumps(reviews))
        return _review_annotations([path], pd.DataFrame(products))


if __name__ == '__main__':
    unittest.main()
