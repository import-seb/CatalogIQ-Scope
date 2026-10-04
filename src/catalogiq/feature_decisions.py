"""Second-pass feature decisions; proposals only, never a final training mask.

Compatibility command: python -m scripts.feature_decisions --help
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path

from .features import (
    IDS, LABELS, NUMBERS, TRIM, clean_row, missing, sha256,
)

POLICY_VERSION = 'feature-decisions-v3'
KEY = ['dataset', 'source_sha256', 'source_row']
EXCLUDED_MODEL_FIELDS = sorted(IDS | LABELS | {
    'MDM_InsertDateTime', 'Exclude', 'Sun1', 'Sun2', 'Sun3', 'Sun4',
    'Sun5', 'Notes',
})
NUMERIC_ERRORS = {'non_numeric', 'non_finite', 'negative_review', 'fractional_count'}


def flag_decision(flag, row, repeated_repaired):
    """Every known flag has an explicit disposition; new flags fail closed."""
    column, kind = flag.split(':', 1)
    category, action = 'requires_review', 'preserve value and field flag; decide row disposition separately'
    if column in NUMBERS and kind in NUMERIC_ERRORS:
        reason = 'Do not coerce text to null or round counts; inspect row alignment.'
    elif flag == 'Upc:scientific_notation_preserved':
        category, action = 'informational', 'preserve string; do not expand or use as a deduplication key'
        reason = 'Original precision and leading zeros cannot be recovered reliably.'
    elif flag == 'Exclude:marked':
        category, action = 'informational', 'retain otherwise usable row under agreed Exclude policy'
        reason = 'Explicit source marker, not evidence of structural corruption.'
    elif flag == 'Exclude:unexpected_value':
        reason = 'Unexpected marker content; never interpret arbitrary text as an exclusion boolean.'
    elif flag == 'MDM_Id:url_in_identifier':
        reason = 'URL in identifier suggests displacement; do not move it to another column.'
    elif flag == 'MDM_InsertDateTime:url_with_displacement':
        reason = 'URL in timestamp plus independent displacement evidence; preserve raw metadata for audit only.'
    elif column in {'ProductUrl', 'ProductImageUrl'} and kind == 'invalid_http_url':
        reason = 'Syntax check only; do not invent URLs or fetch remote content.'
    elif flag == 'ProductName:missing':
        if any(not missing(row.get(c, '')) for c in ('ProductDescription', 'ProductContents')):
            category, action = 'informational', 'preserve missing name; retain available descriptive text'
        else:
            category, action = 'requires_review', 'preserve raw fields; independent structural policy quarantines missing product text'
        reason = 'Do not manufacture product text; missing all three descriptive fields is handled by structural quarantine.'
    elif flag == 'JoiningKey:missing':
        reason = 'Use the source identity for joins; do not generate a business identifier.'
    elif column in {'Sun1', 'Sun2', 'Sun3', 'Sun4', 'Sun5', 'Notes'} and kind == 'populated_review':
        reason = 'Unknown metadata is excluded from model inputs; inspect populated values for displacement.'
    elif flag == 'ProductCategory:repeated_path':
        if repeated_repaired:
            category, action = 'safe_correction', 'collapse exact repeated multi-level breadcrumb'
        reason = 'Only a complete repeated path is eligible; no fuzzy matching or taxonomy inference.'
    elif flag in {'ProductCategory:empty_level', 'ProductCategory:single_level_review'}:
        reason = 'Preserve ambiguous hierarchy; variable depth alone does not justify exclusion.'
    else:
        raise ValueError(f'Unmapped flag: {flag}')
    return dict(flag=flag, category=category, action=action, reason=reason)


def decide_row(row):
    cleaned, flags, _ = clean_row(row)
    bad_numeric = sorted({f.split(':')[0] for f in flags
                          if f.split(':')[0] in NUMBERS and f.split(':')[1] in NUMERIC_ERRORS})
    misplaced = sorted(f for f in flags if f in {
        'MDM_Id:url_in_identifier', 'MDM_InsertDateTime:url_with_displacement',
        'ProductUrl:invalid_http_url', 'ProductImageUrl:invalid_http_url',
    })
    populated_unknown = [f for f in flags if f.endswith(':populated_review')]
    structural_reasons = []
    if 'MDM_InsertDateTime:url_with_displacement' in flags:
        structural_reasons.append('url_in_timestamp_with_displacement')
    # Negative/fractional numbers alone may be field-level quality problems.
    # A structural recommendation needs misplaced text plus another bad anchor.
    text_numeric = [f.split(':')[0] for f in flags
                    if f.split(':')[0] in NUMBERS and f.endswith(':non_numeric')]
    if len(text_numeric) >= 2:
        structural_reasons.append('text_in_multiple_numeric_anchors')
    elif text_numeric and (len(bad_numeric) >= 2 or misplaced):
        structural_reasons.append('text_numeric_failure_with_corroboration')
    if 'Exclude:unexpected_value' in flags and populated_unknown:
        structural_reasons.append('unexpected_exclude_with_populated_unknown_columns')

    # Apply the documented exact-repeat correction only without structural evidence.
    repeated_repaired = False
    if 'ProductCategory:repeated_path' in flags and not structural_reasons:
        parts = cleaned['ProductCategory'].split(' > ')
        for n in range(2, len(parts) // 2 + 1):
            if (len(parts) % n == 0 and len(set(parts[:n])) >= 2
                    and parts == parts[:n] * (len(parts) // n)):
                cleaned['ProductCategory'] = ' > '.join(parts[:n])
                repeated_repaired = True
                break
    events = [flag_decision(f, row, repeated_repaired) for f in flags]
    changes = {c: {'before': row[c], 'after': cleaned[c]}
               for c in row if row[c] != cleaned[c]}
    for column in changes:
        if column != 'ProductCategory' or not repeated_repaired:
            events.append(dict(flag=column + ':format_normalized', category='safe_correction',
                               action='trim outer whitespace or normalize breadcrumb separators',
                               reason='Preserve text meaning and hierarchy depth.'))
    source_marker = 'Exclude:marked' in flags
    hold_reasons = ['missing_product_text'] if any(e['category'] == 'row_hold' for e in events) else []
    requires_review = any(e['category'] in {'requires_review', 'row_hold'} for e in events)
    if structural_reasons:
        disposition = 'quarantine_candidate'
    elif hold_reasons:
        disposition = 'review_hold'
    elif requires_review:
        disposition = 'review'
    elif changes:
        disposition = 'repaired'
    else:
        disposition = 'retain'
    if any(row[c] != cleaned[c] for c in LABELS | IDS if c in row):
        raise ValueError('Protected label or identifier changed.')
    return cleaned, dict(
        disposition=disposition, recommend_quarantine=bool(structural_reasons),
        source_exclude_marker=source_marker, requires_review=requires_review,
        recommend_hold=bool(hold_reasons), hold_reasons=hold_reasons,
        structural_reasons=structural_reasons, numeric_error_columns=bad_numeric,
        corroborating_flags=misplaced + populated_unknown, flags=flags,
        changes=changes, decisions=events,
    )


def run(source, output, dataset):
    if dataset not in {'training', 'target'}:
        raise ValueError('Dataset must be training or target.')
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source.parent) or source.is_relative_to(output):
        raise ValueError('Output must be outside the raw input directory and its ancestors.')
    digest = sha256(source)
    csv.field_size_limit(10_000_000)
    dispositions, flags_count, corrections, rules = Counter(), Counter(), Counter(), Counter()
    marker_counts = Counter()
    total = changed = 0
    with source.open(encoding='utf-8-sig', newline='') as handle, ExitStack() as stack:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames
        required = LABELS | IDS | NUMBERS | TRIM | {
            'ProductCategory', 'MDM_InsertDateTime', 'ProductDescription', 'ProductContents',
            'Exclude', 'Sun1', 'Sun2', 'Sun3', 'Sun4', 'Sun5', 'Notes',
        }
        if not fields or len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError('Missing or duplicate columns; refusing to guess schema.')
        output.mkdir(parents=True, exist_ok=False)
        candidate = csv.DictWriter(stack.enter_context(
            (output / 'features_candidate.csv').open('w', encoding='utf-8', newline='')), fieldnames=fields)
        candidate.writeheader()
        audit = csv.DictWriter(stack.enter_context(
            (output / 'feature_decisions.csv').open('w', encoding='utf-8', newline='')),
            fieldnames=KEY + ['JoiningKey', 'policy_version', 'disposition', 'recommend_quarantine',
                             'source_exclude_marker', 'requires_review', 'recommend_hold', 'hold_reasons', 'reasons', 'flags', 'changed_columns'])
        audit.writeheader()
        detail = stack.enter_context((output / 'decision_details.jsonl').open('w', encoding='utf-8', newline='\n'))
        for index, row in enumerate(reader, 1):
            if None in row or None in row.values():
                raise ValueError(f'Wrong field count at record {index}; incomplete run.')
            cleaned, decision = decide_row(row)
            identity = dict(dataset=dataset, source_sha256=digest, source_row=index)
            candidate.writerow(cleaned)
            audit.writerow({**identity, 'JoiningKey': row['JoiningKey'], 'policy_version': POLICY_VERSION,
                            **{k: decision[k] for k in ('disposition',)},
                            **{k: int(decision[k]) for k in ('recommend_quarantine', 'source_exclude_marker', 'requires_review', 'recommend_hold')},
                            'hold_reasons': ';'.join(decision['hold_reasons']),
                            'reasons': ';'.join(decision['structural_reasons']),
                            'flags': ';'.join(decision['flags']),
                            'changed_columns': ';'.join(sorted(decision['changes']))})
            if decision['decisions'] or decision['structural_reasons']:
                detail.write(json.dumps({**identity, **decision}, sort_keys=True, ensure_ascii=True) + '\n')
            total += 1
            changed += bool(decision['changes'])
            dispositions.update([decision['disposition']])
            flags_count.update(decision['flags'])
            corrections.update(decision['changes'].keys())
            rules.update(decision['structural_reasons'])
            marker_counts['source_exclude_total'] += decision['source_exclude_marker']
            marker_counts['source_exclude_and_quarantine'] += decision['source_exclude_marker'] and decision['recommend_quarantine']
    if sha256(source) != digest:
        raise RuntimeError('Input changed during run; discard outputs.')
    summary = dict(policy_version=POLICY_VERSION, dataset=dataset, source_sha256=digest,
                   rows=total, rows_removed=0, rows_changed=changed, source_row_base=1,
                   dispositions=dict(sorted(dispositions.items())),
                   flags=dict(sorted(flags_count.items())), corrections=dict(sorted(corrections.items())),
                   quarantine_rules=dict(sorted(rules.items())), marker_counts=dict(marker_counts),
                   excluded_model_input_columns=EXCLUDED_MODEL_FIELDS,
                   row_hold_policy='no automatic field-review holds; missing product text is quarantined by the independent structural checker',
                   exclude_policy='keep otherwise usable rows',
                   labels_and_identifiers_preserved=True, final_mask_agreed=False,
                   modeling_ready=False,
                   output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return summary
