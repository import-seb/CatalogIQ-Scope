import csv
import tempfile
import unittest
from pathlib import Path
from scripts.clean_features import clean_row, run, LABELS, IDS, NUMBERS, TRIM


class FeatureCleaningTests(unittest.TestCase):
    def test_preserves_labels_ids_and_suspicious_values(self):
        row = {k: '  null ' for k in LABELS}
        row.update({k: '00123' for k in IDS})
        row.update(Upc='3.00054E+11', ProductRating='ingredient text', ProductName=' Name ',
                   ProductCategory='Health>  Care ', Exclude='Exclude')
        cleaned, flags, _ = clean_row(row)
        for key in LABELS | IDS | {'ProductRating'}:
            self.assertEqual(row[key], cleaned[key])
        self.assertIn('ProductRating:non_numeric', flags)
        self.assertIn('Exclude:marked', flags)
        self.assertEqual(cleaned['ProductCategory'], 'Health > Care')
        self.assertEqual(cleaned['ProductName'], 'Name')
        self.assertEqual(clean_row(cleaned)[0], cleaned)

    def test_nonfinite_fractional_and_repeated_path(self):
        _, flags, _ = clean_row({'ProductRating': 'NaN', 'ProductReviewsCount': '2.5',
                                 'ProductCategory': 'A>B>A>B', 'ProductUrl': 'http://['})
        self.assertIn('ProductRating:non_finite', flags)
        self.assertIn('ProductReviewsCount:fractional_count', flags)
        self.assertIn('ProductCategory:repeated_path', flags)
        self.assertIn('ProductUrl:invalid_http_url', flags)

    def test_file_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.csv'
            fields = sorted(LABELS | IDS | NUMBERS | TRIM | {'ProductCategory', 'MDM_InsertDateTime'})
            row = dict.fromkeys(fields, 'null')
            row.update(ProductName='Quoted "name"\nsecond line', JoiningKey='0001')
            with raw.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(row)
            before = raw.read_bytes()
            out = Path(tmp) / 'output'
            self.assertEqual(run(raw, out, 'training')['rows'], 1)
            with (out / 'features_candidate.csv').open(encoding='utf-8', newline='') as f:
                self.assertEqual(next(csv.DictReader(f)), row)
            self.assertEqual(raw.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                run(raw, out, 'training')


if __name__ == '__main__':
    unittest.main()
