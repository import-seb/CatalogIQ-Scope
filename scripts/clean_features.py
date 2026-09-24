"""Conservative, label-preserving feature cleanup. Python standard library only."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlsplit

LABELS = {'Category', 'Mnfr', 'Brand', 'Platform', 'Segment', 'Sub-Segment', 'TargetAgeGroup'}
IDS = {'JoiningKey', 'Sku', 'Upc', 'ProductModelNumber', 'MDM_Id'}
NUMBERS = {'ProductRating', 'ProductReviewsCount', 'ReviewsCount', 'XRatXRev'}
COUNTS = {'ProductReviewsCount', 'ReviewsCount'}
TRIM = {'Retailer', 'ProductBrand', 'ProductName', 'ProductUrl', 'ProductImageUrl'}
MISSING = {'', 'null'}  # Only observed source tokens; never change target labels.


def missing(value):
    return value.strip().lower() in MISSING


def clean_row(row):
    result = dict(row)
    flags = []
    changes = []
    for column in TRIM:
        if column in row and not missing(row[column]):
            result[column] = row[column].strip()
    category = row.get('ProductCategory', '')
    if not missing(category):
        parts = [part.strip() for part in category.split('>')]
        if any(not part for part in parts):
            flags.append('ProductCategory:empty_level')
        else:
            # Detect repeated paths but leave interpretation to human review.
            if any(len(parts) % n == 0 and parts == parts[:n] * (len(parts) // n)
                   for n in range(1, len(parts) // 2 + 1)):
                flags.append('ProductCategory:repeated_path')
            result['ProductCategory'] = ' > '.join(parts)
        if len(parts) == 1:
            flags.append('ProductCategory:single_level_review')
    for column in NUMBERS:
        value = row.get(column, '')
        if missing(value):
            continue
        try:
            number = Decimal(value.strip())
        except InvalidOperation:
            flags.append(column + ':non_numeric')
            continue
        if not number.is_finite():
            flags.append(column + ':non_finite')
        elif number < 0:
            flags.append(column + ':negative_review')
        elif column in COUNTS and number != number.to_integral_value():
            flags.append(column + ':fractional_count')
        # Keep source numeric strings. Casting requires downstream schema agreement.
    for column in ('ProductUrl', 'ProductImageUrl'):
        value = row.get(column, '')
        if not missing(value):
            try:
                parsed = urlsplit(value.strip())
                valid = parsed.scheme in ('http', 'https') and bool(parsed.hostname)
            except ValueError:
                valid = False
            if not valid:
                flags.append(column + ':invalid_http_url')
    for column in ('ProductName', 'JoiningKey'):
        if missing(row.get(column, '')):
            flags.append(column + ':missing')
    for column in ('Sun1', 'Sun2', 'Sun3', 'Sun4', 'Sun5', 'Notes'):
        if not missing(row.get(column, '')):
            flags.append(column + ':populated_review')
    if not missing(row.get('Exclude', '')):
        flags.append('Exclude:marked' if row['Exclude'].strip() == 'Exclude'
                     else 'Exclude:unexpected_value')
    upc = row.get('Upc', '')
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?[eE][+-]?\d+', upc.strip()):
        flags.append('Upc:scientific_notation_preserved')
    mdm = row.get('MDM_Id', '')
    if mdm.strip().lower().startswith(('http://', 'https://')):
        flags.append('MDM_Id:url_in_identifier')
    stamp = row.get('MDM_InsertDateTime', '')
    if not missing(stamp):
        try:
            number = Decimal(stamp.strip())
            if not number.is_finite():
                raise InvalidOperation
        except InvalidOperation:
            flags.append('MDM_InsertDateTime:non_numeric_review')
    for column in row:
        if result[column] != row[column]:
            changes.append(column)
    assert all(result[k] == row[k] for k in LABELS | IDS if k in row)
    return result, sorted(flags), changes


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(source, output, dataset):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError('Output must be a separate directory outside the raw input directory.')
    digest = sha256(source)
    csv.field_size_limit(10_000_000)
    flags_count, changes_count, blanks = Counter(), Counter(), Counter()
    seen_keys = Counter()
    total = flagged = changed = 0
    with source.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames
        required = LABELS | IDS | NUMBERS | TRIM | {'ProductCategory', 'MDM_InsertDateTime'}
        if not fields or len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError('Missing or duplicate columns; refusing to guess schema.')
        # Refuse accidental overwrites, including interrupted runs.
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'features_candidate.csv').open('w', encoding='utf-8', newline='') as out, \
             (output / 'row_audit.csv').open('w', encoding='utf-8', newline='') as audit:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            audit_writer = csv.writer(audit)
            audit_writer.writerow(['dataset', 'source_sha256', 'source_row', 'JoiningKey', 'flags', 'changed_columns'])
            for index, row in enumerate(reader, 1):
                if None in row or None in row.values():
                    raise ValueError(f'Wrong field count at data record {index}; run incomplete.')
                cleaned, flags, changes = clean_row(row)
                key = row['JoiningKey']
                if not missing(key):
                    seen_keys[key] += 1
                for column, value in row.items():
                    if missing(value):
                        blanks[column] += 1
                writer.writerow(cleaned)
                audit_writer.writerow([dataset, digest, index, key, ';'.join(flags), ';'.join(changes)])
                total += 1
                flagged += bool(flags)
                changed += bool(changes)
                flags_count.update(flags)
                changes_count.update(changes)
    if sha256(source) != digest:
        raise RuntimeError('Input changed during processing; discard this run.')
    summary = dict(dataset=dataset, source_sha256=digest, rows=total, columns=len(fields),
                   rows_with_review_flags=flagged, rows_changed=changed, rows_removed=0,
                   duplicate_joiningkey_groups=sum(n > 1 for n in seen_keys.values()),
                   flags=dict(sorted(flags_count.items())), changes=dict(sorted(changes_count.items())),
                   missing_before_and_after=dict(sorted(blanks.items())),
                   labels_and_identifiers_preserved=True,
                   candidate_sha256=sha256(output / 'features_candidate.csv'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', required=True, choices=['training', 'target'])
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.dataset), indent=2))
