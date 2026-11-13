export const FIELDS = ['ProductCategory', 'ProductName', 'ProductBrand', 'ProductDescription', 'ProductContents'];
export const TARGETS = ['Segment', 'Sub-Segment', 'Brand', 'Mnfr', 'TargetAgeGroup', 'Platform'];
export const TITLES = { Segment: 'Segment', 'Sub-Segment': 'Sub-segment', Brand: 'Brand', Mnfr: 'Manufacturer', TargetAgeGroup: 'Age group', Platform: 'Platform' };
export const MAX_ROWS = 250;
export const MAX_BYTES = 2 * 1024 * 1024;
export const SCHEMA = 'catalogiq-inference-request-v1';

// RFC 4180-style quoted cells, embedded newlines, BOM, CRLF and explicit malformed-file errors.
export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = []; let row = [], cell = '', quoted = false, closed = false;
  const finishCell = () => { row.push(cell); cell = ''; closed = false; };
  const finishRow = () => { finishCell(); if (row.some(v => v !== '')) rows.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') { quoted = false; closed = true; }
      else cell += c;
    } else if (c === ',') finishCell();
    else if (c === '\r' || c === '\n') { if (c === '\r' && text[i + 1] === '\n') i++; finishRow(); }
    else if (closed) throw new Error('Unexpected text after a closing quote. Please check your CSV file.');
    else if (c === '"') { if (cell) throw new Error('A quote appears inside an unquoted cell. Please check your CSV file.'); quoted = true; }
    else cell += c;
  }
  if (quoted) throw new Error('A quoted cell is not closed. Please check your CSV file.');
  if (cell || row.length || closed) finishRow();
  if (rows.length < 2) throw new Error('Add a header row and at least one product.');
  const headers = rows.shift();
  if (headers.some(h => !h.trim()) || new Set(headers).size !== headers.length) throw new Error('Column names must be nonempty and unique.');
  if (rows.length > MAX_ROWS) throw new Error(`Choose up to ${MAX_ROWS} products at a time.`);
  for (const field of ['record_id', ...FIELDS]) if (!headers.includes(field)) throw new Error(`Missing column: ${field}. Download the CSV template for the required columns.`);
  const ids = new Set();
  return rows.map((cells, index) => {
    if (cells.length !== headers.length) throw new Error(`Row ${index + 2} has ${cells.length} cells; expected ${headers.length}.`);
    const source = Object.fromEntries(headers.map((key, i) => [key, cells[i]]));
    const id = source.record_id;
    if (!id.trim() || ids.has(id)) throw new Error(`Row ${index + 2}: product IDs must be nonempty and unique.`);
    ids.add(id);
    const product = Object.fromEntries(['record_id', ...FIELDS].map(f => [f, source[f]]));
    if (Object.hasOwn(source, 'quarantined')) {
      if (!['', 'true', 'false'].includes(source.quarantined.toLowerCase())) throw new Error(`Row ${index + 2}: quarantined must be true or false.`);
      product.quarantined = source.quarantined.toLowerCase() === 'true';
    }
    return { product, source };
  });
}

export function alignResponse(response, inputs) {
  if (!response || response.schema_version !== SCHEMA || !Array.isArray(response.products) || response.products.length !== inputs.length) throw new Error('The response does not match this batch. No results were applied.');
  const map = new Map();
  for (const row of response.products) {
    if (map.has(row.record_id) || !inputs.some(p => p.record_id === row.record_id)) throw new Error('The response contains unexpected or duplicate product IDs.');
    for (const target of TARGETS) {
      const p = row.predictions?.[target];
      if (!p || !['suggestion', 'failed', 'unavailable'].includes(p.status) || typeof p.review_required !== 'boolean' || !Array.isArray(p.review_reasons)) throw new Error('The response contains an incomplete prediction.');
      if (p.status === 'suggestion' && (typeof p.predicted_label !== 'string' || !Number.isFinite(p.confidence) || p.confidence < 0 || p.confidence > 1)) throw new Error('The response contains an invalid confidence or label.');
    }
    map.set(row.record_id, row);
  }
  return inputs.map(input => map.get(input.record_id));
}

export function alternatives(prediction, limit = 3) {
  return Object.entries(prediction?.probabilities || {}).map(([key, probability]) => ({label: prediction.label_mapping?.[key], probability}))
    .filter(v => typeof v.label === 'string' && Number.isFinite(v.probability))
    .sort((a,b) => b.probability - a.probability).slice(0, limit);
}

export function canReview(record) {
  return TARGETS.filter(t => t !== 'Platform').every(t => record.result?.predictions[t]?.status === 'suggestion');
}

export function recordDecision(record, action, labels, note, reviewer, now = new Date().toISOString()) {
  if (!reviewer.trim()) throw new Error('Enter your name to save a review.');
  if (!['approved', 'rejected', 'corrected'].includes(action)) throw new Error('Choose a review action.');
  if (action !== 'rejected' && !canReview(record)) throw new Error('Predictions must be available before you can approve them.');
  const changed = Object.keys(labels).some(t => labels[t] !== record.result?.predictions[t]?.predicted_label);
  if (action === 'corrected' && !changed) throw new Error('Change a prediction before saving a correction.');
  if (action === 'approved' && changed) throw new Error('Save the correction to keep your changed labels.');
  const expected = TARGETS.filter(t => record.result?.predictions[t]?.status === 'suggestion');
  if (action !== 'rejected' && (Object.keys(labels).length !== expected.length || expected.some(t => !Object.hasOwn(labels,t)))) throw new Error('Review every available target before saving.');
  if ((action === 'corrected' || action === 'rejected') && !note.trim()) throw new Error('Add a short reason for this decision.');
  for (const [target, label] of Object.entries(labels)) {
    const prediction = record.result?.predictions[target];
    if (!prediction || prediction.status !== 'suggestion' || !Object.values(prediction.label_mapping || {}).includes(label)) throw new Error('Choose a label from the model’s available labels.');
  }
  const decision = {action, labels: action === 'rejected' ? {} : {...labels}, note: note.trim(), reviewer: reviewer.trim(), timestamp: now};
  return {...record, decision, history: [...(record.history || []), decision]};
}

export function csvCell(value) {
  let text = value == null ? '' : String(value);
  // Neutralize spreadsheet formulas; retain exact raw fields separately in the JSON audit export.
  if (/^[\s]*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text)) text = "'" + text;
  return '"' + text.replaceAll('"', '""') + '"';
}

export function exportCSV(records) {
  const sourceKeys = [...new Set(records.flatMap(r => Object.keys(r.source)))];
  const generated = ['catalogiq:mode','catalogiq:review_status','catalogiq:reviewer','catalogiq:reviewed_at','catalogiq:review_note','catalogiq:engine_version','catalogiq:registry_version'];
  for (const t of TARGETS) generated.push(...['original_label','final_label','raw_confidence','prediction_status','review_reasons','model_version'].map(k=>`catalogiq:${t}:${k}`));
  const headers = [...sourceKeys.map(k => 'source:' + k), ...generated];
  const lines = records.map(r => {
    const d = r.decision;
    const cells = [...sourceKeys.map(k => r.source[k]),r.mode,d?.action || 'pending',d?.reviewer,d?.timestamp,d?.note,r.engine_version,r.relationship_registry_version];
    for (const target of TARGETS) {
      const p = r.result?.predictions[target];
      cells.push(p?.predicted_label, d && d.action !== 'rejected' ? d.labels[target] : '', p?.confidence, p?.status || 'not_run', p?.review_reasons?.join('; '),p?.provenance?.model_id);
    }
    return cells.map(csvCell).join(',');
  });
  return '\uFEFF' + [headers.map(csvCell).join(','), ...lines].join('\r\n');
}

export function auditExport(records) {
  return JSON.stringify({schema_version:'catalogiq-review-export-v1', exported_at:new Date().toISOString(),records},null,2);
}
