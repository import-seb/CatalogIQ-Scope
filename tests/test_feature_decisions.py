import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.clean_features import LABELS, IDS, NUMBERS, TRIM
from scripts.feature_decisions import decide_row, flag_decision, run
from scripts.validate_feature_decisions import verify, valid_category_change


def sample_row(**overrides):
    fields = LABELS | IDS | NUMBERS | TRIM | {
        'ProductCategory', 'MDM_InsertDateTime', 'ProductDescription', 'ProductContents',
        'Exclude', 'Sun1', 'Sun2', 'Sun3', 'Sun4', 'Sun5', 'Notes',
    }
    row = dict.fromkeys(sorted(fields), 'null')
    row.update(ProductName='Sample product', JoiningKey='0001', ProductCategory='Health > Care')
    row.update(overrides)
    return row


class FeatureDecisionTests(unittest.TestCase):
    def test_blank_category_is_preserved_without_category_flags(self):
        for value in ('', '   ', '\t'):
            with self.subTest(value=value):
                row = sample_row(ProductCategory=value)
                cleaned, decision = decide_row(row)
                self.assertEqual(cleaned, row)
                self.assertFalse(any(f.startswith('ProductCategory:') for f in decision['flags']))
                self.assertTrue(valid_category_change(value, value))
                self.assertFalse(valid_category_change(value, 'Invented category'))

    def test_null_category_is_preserved_without_category_flags(self):
        for value in (None, 'null', ' NULL ', 'Null'):
            with self.subTest(value=value):
                row = sample_row(ProductCategory=value)
                cleaned, decision = decide_row(row)
                self.assertEqual(cleaned, row)
                self.assertFalse(any(f.startswith('ProductCategory:') for f in decision['flags']))
                self.assertTrue(valid_category_change(value, value))
                self.assertFalse(valid_category_change(value, 'Invented category'))
        self.assertFalse(valid_category_change(' null ', 'null'))

    def test_scientific_upc_retained_without_quarantine(self):
        raw = sample_row(Upc='3.00054E+11', Platform=' Rare label ', MDM_Id='000045')
        cleaned, decision = decide_row(raw)
        self.assertEqual(decision['disposition'], 'retain')
        self.assertFalse(decision['recommend_quarantine'])
        for col in IDS | LABELS:
            self.assertEqual(cleaned[col], raw[col])

    def test_source_marker_is_separate_policy_not_corruption(self):
        _, decision = decide_row(sample_row(Exclude='Exclude'))
        self.assertEqual(decision['disposition'], 'retain')
        self.assertTrue(decision['source_exclude_marker'])
        self.assertFalse(decision['recommend_quarantine'])
        _, unexpected = decide_row(sample_row(Exclude='other text'))
        self.assertEqual(unexpected['disposition'], 'review')
        self.assertFalse(unexpected['source_exclude_marker'])

    def test_single_failure_or_two_fractional_counts_are_not_structural(self):
        for values in ({'ProductRating': 'description'},
                       {'ReviewsCount': '2.5', 'ProductReviewsCount': '4.2'},
                       {'ProductRating': 'NaN'}, {'ReviewsCount': '-1'}):
            with self.subTest(values=values):
                raw = sample_row(**values)
                cleaned, decision = decide_row(raw)
                self.assertEqual(cleaned, raw)
                self.assertEqual(decision['disposition'], 'review')
                self.assertFalse(decision['recommend_quarantine'])

    def test_text_with_corroborating_anchor_recommends_quarantine(self):
        cases = [dict(ProductRating='description', ProductReviewsCount='fragment'),
                 dict(ProductRating='description', ReviewsCount='3.5'),
                 dict(ProductRating='description', MDM_Id='https://example.test/image')]
        for values in cases:
            with self.subTest(values=values):
                raw = sample_row(**values)
                cleaned, decision = decide_row(raw)
                self.assertEqual(cleaned, raw)
                self.assertEqual(decision['disposition'], 'quarantine_candidate')
                self.assertTrue(decision['structural_reasons'])

    def test_unknown_metadata_needs_corroboration(self):
        _, alone = decide_row(sample_row(Sun1='ingredient'))
        self.assertFalse(alone['recommend_quarantine'])
        _, corroborated = decide_row(sample_row(Sun1='ingredient', Exclude='shifted text'))
        self.assertTrue(corroborated['recommend_quarantine'])

    def test_exact_path_repair_and_ambiguous_paths(self):
        raw = sample_row(ProductCategory=' A>B > A >B ')
        clean, decision = decide_row(raw)
        self.assertEqual(clean['ProductCategory'], 'A > B')
        self.assertEqual(decision['disposition'], 'repaired')
        self.assertEqual(decision['changes']['ProductCategory']['before'], raw['ProductCategory'])
        self.assertEqual(decide_row(clean)[0], clean)
        for path in ('A > A', 'A > > B', 'A > B > A > C'):
            clean, decision = decide_row(sample_row(ProductCategory=path))
            self.assertEqual(clean['ProductCategory'], path)
        raw = sample_row(ProductCategory='A > B > A > B', ProductRating='bad', ReviewsCount='text')
        self.assertEqual(decide_row(raw)[0]['ProductCategory'], raw['ProductCategory'])

    def test_missing_name_uses_available_text_without_inventing_values(self):
        raw = sample_row(ProductName='null', ProductDescription='Some description')
        clean, decision = decide_row(raw)
        self.assertEqual(clean, raw)
        self.assertEqual(decision['disposition'], 'retain')
        _, empty = decide_row(sample_row(ProductName='null'))
        self.assertEqual(empty['disposition'], 'review')
        self.assertFalse(empty['recommend_hold'])
        self.assertFalse(empty['recommend_quarantine'])

    def test_unmapped_flag_cannot_silently_pass(self):
        with self.assertRaisesRegex(ValueError, 'Unmapped flag'):
            flag_decision('new:flag', {}, False)

    def test_full_run_is_repeatable_keeps_rows_and_uses_record_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'raw').mkdir()
            source = root / 'raw' / 'data.csv'
            rows = [sample_row(ProductName=' line one\nline two ', JoiningKey='duplicate'),
                    sample_row(JoiningKey='duplicate', ProductRating='bad', ReviewsCount='text'),
                    sample_row(Exclude='Exclude', Upc='3.0E+11')]
            with source.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            first = run(source, root / 'out1', 'target')
            second = run(source, root / 'out2', 'target')
            self.assertEqual(first, second)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(first['rows'], 3)
            self.assertEqual(first['rows_removed'], 0)
            self.assertFalse(first['modeling_ready'])
            self.assertEqual(verify(source, root / 'out1', 'target')['verified_rows'], 3)
            with (root / 'out1' / 'feature_decisions.csv').open(newline='') as f:
                audit = list(csv.DictReader(f))
            self.assertEqual([r['source_row'] for r in audit], ['1', '2', '3'])
            self.assertEqual([r['dataset'] for r in audit], ['target'] * 3)
            self.assertEqual([r['disposition'] for r in audit], ['repaired', 'quarantine_candidate', 'retain'])
            with (root / 'out1' / 'features_candidate.csv').open(newline='') as f:
                cleaned = list(csv.DictReader(f))
            self.assertEqual(len(cleaned), len(rows))
            for old, new in zip(rows, cleaned):
                for col in LABELS | IDS:
                    self.assertEqual(old[col], new[col])
            for filename in first['output_sha256']:
                self.assertEqual((root / 'out1' / filename).read_bytes(), (root / 'out2' / filename).read_bytes())
            with self.assertRaises(FileExistsError):
                run(source, root / 'out1', 'target')
            for unsafe in (source, source.parent, source.parent / 'out', root):
                with self.assertRaises(ValueError):
                    run(source, unsafe, 'target')
            audit[0]['source_row'] = '0'
            audit_path = root / 'out1' / 'feature_decisions.csv'
            with audit_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(audit[0]))
                writer.writeheader()
                writer.writerows(audit)
            # Rehash the tampered file to ensure the row contract is checked too.
            from scripts.clean_features import sha256
            first['output_sha256']['feature_decisions.csv'] = sha256(audit_path)
            (root / 'out1' / 'summary.json').write_text(json.dumps(first), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid identity/order'):
                verify(source, root / 'out1', 'target')

    def test_bad_schema_or_row_never_gets_completion_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'raw').mkdir()
            source = root / 'raw' / 'data.csv'
            source.write_text('name,name\nx,x\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                run(source, root / 'bad_schema', 'training')
            self.assertFalse((root / 'bad_schema').exists())
            row = sample_row()
            with source.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row.keys())
                writer.writerow(['too', 'few'])
            with self.assertRaises(ValueError):
                run(source, root / 'bad_row', 'training')
            self.assertFalse((root / 'bad_row' / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
