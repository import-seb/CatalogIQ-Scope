"""Verify second-pass outputs against raw data without calling the cleaner."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from itertools import zip_longest
import json
from pathlib import Path

from .features import IDS, LABELS, TRIM, missing, sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_category_change(before, after):
    if missing(before) or missing(after):
        return before == after
    parts = [p.strip() for p in before.split('>')]
    if not all(parts):
        return before == after
    normalized = ' > '.join(parts)
    if normalized == after:
        return True
    shorter = after.split(' > ')
    return (len(shorter) >= 2 and len(set(shorter)) >= 2 and len(shorter) < len(parts)
            and len(parts) % len(shorter) == 0
            and parts == shorter * (len(parts) // len(shorter)))


def verify(source, output, dataset):
    source, output = Path(source), Path(output)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    digest = sha256(source)
    require(summary['dataset'] == dataset, 'Dataset mismatch.')
    require(summary['source_sha256'] == digest, 'Raw input hash mismatch.')
    require(summary['source_row_base'] == 1, 'Source record numbering must start at 1.')
    expected_files = {'features_candidate.csv', 'feature_decisions.csv', 'decision_details.jsonl'}
    require(set(summary['output_sha256']) == expected_files, 'Unexpected output manifest.')
    for name, expected in summary['output_sha256'].items():
        require(sha256(output / name) == expected, f'Output hash mismatch: {name}')
    details = {}
    with (output / 'decision_details.jsonl').open(encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            index = record['source_row']
            require(index not in details, 'Duplicate detail record.')
            require(record['dataset'] == dataset and record['source_sha256'] == digest,
                    'Detail source identity mismatch.')
            details[index] = record
    dispositions, changes_count, flags_count, reasons_count = Counter(), Counter(), Counter(), Counter()
    rows = changed_rows = exclude_count = overlap_count = 0
    csv.field_size_limit(10_000_000)
    with ExitStack() as stack:
        readers = [csv.DictReader(stack.enter_context(p.open(encoding='utf-8-sig', newline='')), strict=True)
                   for p in (source, output / 'features_candidate.csv', output / 'feature_decisions.csv')]
        require(readers[0].fieldnames == readers[1].fieldnames, 'Source schema changed.')
        for index, triple in enumerate(zip_longest(*readers), 1):
            raw, candidate, audit = triple
            require(all(r is not None for r in triple), 'Row count mismatch.')
            require(all(None not in r and None not in r.values() for r in triple), 'Malformed CSV row.')
            require(audit['dataset'] == dataset and audit['source_sha256'] == digest
                    and audit['source_row'] == str(index), f'Invalid identity/order at record {index}.')
            require(audit['JoiningKey'] == raw['JoiningKey'], 'JoiningKey audit mismatch.')
            require(all(raw[c] == candidate[c] for c in IDS | LABELS), 'Protected value changed.')
            actual_changes = {c: {'before': raw[c], 'after': candidate[c]}
                              for c in raw if raw[c] != candidate[c]}
            require(set(actual_changes) <= TRIM | {'ProductCategory'}, 'Unexpected changed column.')
            for col in actual_changes:
                valid = (candidate[col] == raw[col].strip() if col in TRIM
                         else valid_category_change(raw[col], candidate[col]))
                require(valid, f'Non-whitelisted correction at record {index}, {col}.')
            require(audit['changed_columns'] == ';'.join(sorted(actual_changes)), 'Change list mismatch.')
            flags = audit['flags'].split(';') if audit['flags'] else []
            reasons = audit['reasons'].split(';') if audit['reasons'] else []
            for name in ('recommend_quarantine', 'source_exclude_marker', 'requires_review'):
                require(audit[name] in {'0', '1'}, f'Invalid boolean: {name}')
            quarantine = audit['recommend_quarantine'] == '1'
            excluded = raw['Exclude'].strip() == 'Exclude'
            require((audit['source_exclude_marker'] == '1') == excluded, 'Exclude interpretation changed.')
            require(quarantine == bool(reasons), 'Quarantine reason mismatch.')
            expected_disposition = ('quarantine_candidate' if quarantine else 'policy_hold' if excluded
                                    else 'review' if audit['requires_review'] == '1'
                                    else 'repaired' if actual_changes else 'retain')
            require(audit['disposition'] == expected_disposition, 'Disposition mismatch.')
            detail = details.pop(index, None)
            require(bool(detail) == bool(flags or actual_changes or reasons), 'Detail coverage mismatch.')
            if detail:
                require(detail['changes'] == actual_changes and detail['flags'] == flags,
                        'Detailed audit does not match output changes/flags.')
                require(detail['structural_reasons'] == reasons
                        and detail['disposition'] == audit['disposition'], 'Detail decision mismatch.')
                for name in ('recommend_quarantine', 'source_exclude_marker', 'requires_review'):
                    require(detail[name] == (audit[name] == '1'), 'Detail boolean mismatch.')
                require(set(flags) <= {e['flag'] for e in detail['decisions']}, 'Unresolved flag in detail.')
            dispositions.update([audit['disposition']])
            changes_count.update(actual_changes.keys())
            flags_count.update(flags)
            reasons_count.update(reasons)
            rows += 1
            changed_rows += bool(actual_changes)
            exclude_count += excluded
            overlap_count += excluded and quarantine
    require(not details, 'Extra detail records.')
    for name, actual in [('rows', rows), ('rows_changed', changed_rows), ('rows_removed', 0),
                         ('dispositions', dict(dispositions)), ('corrections', dict(changes_count)),
                         ('flags', dict(flags_count)), ('quarantine_rules', dict(reasons_count)),
                         ('marker_counts', {'source_exclude_total': exclude_count,
                                            'source_exclude_and_quarantine': overlap_count})]:
        require(summary[name] == actual, f'Summary mismatch: {name}')
    require(summary['modeling_ready'] is False and summary['final_mask_agreed'] is False,
            'Premature readiness claim.')
    require(sha256(source) == digest, 'Source changed during verification.')
    return dict(dataset=dataset, verified_rows=rows, source_sha256=digest,
                raw_and_protected_values_preserved=True, full_row_order_and_key_check=True,
                corrections_and_audit_reconciled=True, summaries_reconciled=True,
                output_hashes_verified=True, modeling_ready=False)
