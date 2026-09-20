"""Local-only Aryx smoke test; saves record-level evidence in ignored data/derived."""
import argparse
import csv
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE = 'http://127.0.0.1:8088'
ROOT = Path(__file__).resolve().parents[1] / 'data/derived/aryx-sample'
STATE = ROOT / 'validation.json'


def call(path, body=None, content_type='application/json', timeout=180):
    if isinstance(body, dict):
        body = json.dumps(body).encode()
    req = Request(BASE + path, data=body)
    if body is not None:
        req.add_header('Content-Type', content_type)
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def save(state):
    STATE.write_text(json.dumps(state, indent=2), encoding='utf-8')


def prepare():
    from clean_features import clean_row, sha256
    if STATE.exists():
        raise SystemExit('Existing run: do not replace its input sample.')
    ROOT.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10_000_000)
    rows, manifest, hashes = [], [], {}
    for tag, title in [('training', 'Training'), ('target', 'Target')]:
        path = ROOT.parents[1] / 'provided' / f'Selfcare_{title}_data.csv'
        hashes[tag] = sha256(path)
        normal = bad = 0
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for index, row in enumerate(csv.DictReader(handle), 1):
                _, flags, _ = clean_row(row)
                structural = any(':non_numeric' in f or ':url_in_identifier' in f for f in flags)
                if (structural and bad < 5) or (not flags and normal < 5):
                    bad += int(structural)
                    normal += int(not flags)
                    identity = f'{tag}:{index}'
                    record = dict(name=identity, source_dataset=tag, source_row=str(index))
                    record.update({k: row[k] for k in ['ProductName', 'ProductBrand',
                                   'ProductCategory', 'Retailer', 'JoiningKey']})
                    rows.append(record)
                    manifest.append(dict(name=identity, selection='structural-review' if structural
                                         else 'no-current-flags', flags=flags))
                if bad == normal == 5:
                    break
        if bad != 5 or normal != 5:
            raise ValueError('Insufficient records for the specified sample.')
    with (ROOT / 'catalogiq_features_sample.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'selection.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (ROOT / 'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print('Prepared 20 records; no data uploaded.')


def start():
    if STATE.exists():
        raise SystemExit('Existing validation state: inspect it before creating another workspace.')
    health = call('/llm/health')
    if not health.get('ok') or health.get('provider') != 'ollama':
        raise SystemExit('A ready local Ollama provider is required.')
    sample = ROOT / 'catalogiq_features_sample.csv'
    data = sample.read_bytes()
    ws = call('/admin/workspaces', {'name': 'CatalogIQ feature validation',
        'description': '20 sampled source records; local validation only, not model training.'})
    state = {'workspace_id': ws['id'], 'llm_health': health}
    save(state)
    boundary = 'CatalogIQValidationBoundary'
    parts = []
    for key, value in [('workspace_id', str(ws['id'])), ('ontology_type', 'CatalogProduct'),
                       ('match_keys', 'name')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{sample.name}"\r\nContent-Type: text/csv\r\n\r\n'.encode() + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    state['ingest'] = call('/admin/ingest/file', b''.join(parts), f'multipart/form-data; boundary={boundary}')
    save(state)
    print(json.dumps(state))


def check():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    wid = state['workspace_id']
    state['jobs'] = call(f'/admin/jobs?workspace_id={wid}')
    graph = call(f'/graph?workspace_id={wid}')
    state['graph'] = graph
    details = [call(f'/data/entity/{e["id"]}?workspace_id={wid}') for e in graph['entities']]
    provenance = [call(f'/entities/{e["id"]}/provenance?workspace_id={wid}') for e in graph['entities']]
    state['details'], state['provenance'] = details, provenance
    with (ROOT / 'catalogiq_features_sample.csv').open(encoding='utf-8', newline='') as f:
        sample = list(csv.DictReader(f))
    expected = {r['name'] for r in sample}
    observed = {e['name'] for e in graph['entities']}
    by_name = {e['name']: e.get('attributes', {}) for e in details}
    mismatches = [{'name': row['name'], 'column': key}
                  for row in sample for key, value in row.items()
                  if by_name.get(row['name'], {}).get(key) != value]
    state['attribute_mismatches'] = mismatches
    state['checks'] = {'expected_records': len(sample), 'entities': len(details),
        'relationships': len(graph['relationships']), 'names_exact': expected == observed,
        'all_entities_have_provenance': bool(provenance) and all(provenance),
        'attribute_mismatch_count': len(mismatches)}
    save(state)
    print(json.dumps({'jobs': state['jobs'], 'checks': state['checks']}))


def ask():
    state = json.loads(STATE.read_text(encoding='utf-8'))
    with (ROOT / 'catalogiq_features_sample.csv').open(encoding='utf-8', newline='') as f:
        first = next(csv.DictReader(f))
    question = f'What retailer is recorded for {first["name"]}? Answer in one short sentence with a source citation. Do not infer missing values.'
    result = call('/ask', {'workspace_id': state['workspace_id'], 'question': question})
    state['ask'] = {'question': question, 'expected_retailer': first['Retailer'], 'response': result}
    save(state)
    # Keep product content and generated answer in the ignored evidence file.
    print(json.dumps({'answer_chars': len(result.get('answer', '')),
                      'usage': result.get('usage'), 'grounding': result.get('grounding')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'start', 'check', 'ask'])
    globals()[parser.parse_args().action]()
