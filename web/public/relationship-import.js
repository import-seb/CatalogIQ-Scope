import { FIELDS, TARGETS, TITLES, SCHEMA, readCSVTable } from './domain.js';

export const RESULT_FILE_BYTES = 64 * 1024 * 1024;
export const ELIGIBILITY_VERSION = 'target-eligibility-20261116-v2';
export const RELATIONSHIP_MESSAGES = {
  invalid_approved_classification_relationship: 'These classifications may not belong together. Check the product before confirming or correcting them.',
  missing_relationship_classification: 'A classification needed to check this relationship is missing. Check the available product information.',
};

const WARNING_COLUMNS = ['record_id', 'relation_id', 'observation', 'child_target',
  'child_prediction', 'child_confidence', 'parent_target', 'parent_prediction',
  'parent_confidence', 'relation_status', 'registry_version', 'is_approved_decision'];

function requireColumns(headers, required, file) {
  const missing = required.filter(name => !headers.includes(name));
  if (missing.length) throw new Error(`${file} is missing ${missing.join(', ')}.`);
}

function score(value, context) {
  if (value === '') return null;
  const number = Number(value);
  if (!value.trim() || !Number.isFinite(number) || number < 0 || number > 1) {
    throw new Error(`Invalid confidence in ${context}.`);
  }
  return number;
}

function boolean(value, context) {
  if (!['true', 'false'].includes(value?.toLowerCase())) throw new Error(`Invalid review flag in ${context}.`);
  return value.toLowerCase() === 'true';
}

export function warningMessage(warning) {
  const child = TITLES[warning.child_target] || warning.child_target;
  const parent = TITLES[warning.parent_target] || warning.parent_target;
  return `${child} / ${parent}. ${RELATIONSHIP_MESSAGES[warning.observation]}`;
}

// Maria's CSV contains scores and labels, but no full probability distributions.
// Reuse only the vocabulary of the exact same frozen model in the saved envelope.
export function parseSavedResults(predictionsText, warningsText, report, reference, registry) {
  if (report?.eligibility_version !== ELIGIBILITY_VERSION) {
    throw new Error(`Use the corrected results with eligibility version ${ELIGIBILITY_VERSION}.`);
  }
  if (report.schema_version !== SCHEMA || report.engine_version !== reference.engine_version ||
      report.relationship_registry_version !== reference.relationship_registry_version ||
      registry?.registry_version !== report.relationship_registry_version || !Array.isArray(registry.relations) ||
      !/^[a-f0-9]{64}$/.test(report.decisions_sha256 || '')) {
    throw new Error('The report does not match the supported model and cleaning versions.');
  }
  if (!Number.isSafeInteger(report.products_written) || report.products_written < 1 ||
      !Number.isSafeInteger(report.ineligible) || report.ineligible < 0 ||
      report.products_written !== report.eligible || report.eligible + report.ineligible !== report.rows_in_file) {
    throw new Error('The report has inconsistent product counts.');
  }
  const { headers, records: rows } = readCSVTable(predictionsText);
  const targetColumns = TARGETS.flatMap(target => ['original_label', 'final_label', 'raw_confidence',
    'calibrated_confidence', 'prediction_status', 'selection_status', 'review_required',
    'review_reasons', 'model_version'].map(field => `catalogiq:${target}:${field}`));
  requireColumns(headers, ['JoiningKey', 'catalogiq:eligibility', 'catalogiq:eligibility_version',
    'catalogiq:schema_version', 'catalogiq:engine_version', 'catalogiq:registry_version',
    'catalogiq:review_status', 'catalogiq:reviewer', 'catalogiq:reviewed_at', 'catalogiq:review_note',
    'catalogiq:review_required', ...targetColumns], 'Predictions CSV');
  if (rows.length !== report.products_written) throw new Error('Predictions row count does not match the report.');

  const byId = new Map();
  const statuses = Object.fromEntries(TARGETS.map(target => [target, {}]));
  const records = rows.map(row => {
    const id = row.JoiningKey;
    if (!id.trim() || byId.has(id)) throw new Error('Saved product IDs must be nonempty and unique.');
    for (const [field, expected] of Object.entries({eligibility: 'eligible', eligibility_version: ELIGIBILITY_VERSION,
      schema_version: SCHEMA, engine_version: report.engine_version, registry_version: report.relationship_registry_version})) {
      if (row[`catalogiq:${field}`] !== expected) throw new Error(`Product ${id} has a different ${field}.`);
    }
    if (['review_status', 'reviewer', 'reviewed_at', 'review_note'].some(field => row[`catalogiq:${field}`] !== '')) {
      throw new Error('Choose the original prediction export with empty review fields.');
    }
    if (!boolean(row['catalogiq:review_required'], id)) throw new Error('Saved results must require human review.');
    const predictions = {};
    for (const target of TARGETS) {
      const prefix = `catalogiq:${target}:`;
      const template = reference.products[0].predictions[target];
      const status = row[prefix + 'prediction_status'];
      const label = row[prefix + 'original_label'];
      const modelId = row[prefix + 'model_version'];
      const confidence = score(row[prefix + 'raw_confidence'], id);
      if (!['suggestion', 'unavailable', 'failed'].includes(status) || row[prefix + 'final_label'] !== '' ||
          !boolean(row[prefix + 'review_required'], id)) throw new Error(`Invalid saved prediction for ${id}, ${target}.`);
      if (status === 'suggestion' && (confidence === null || !Object.values(template.label_mapping).includes(label) ||
          modelId !== template.provenance.model_id || target === 'Platform')) {
        throw new Error(`The label, model or confidence for ${id}, ${target} does not match the frozen model.`);
      }
      if (status !== 'suggestion' && (label !== '' || confidence !== null)) {
        throw new Error(`Unavailable predictions for ${id}, ${target} must have empty labels and scores.`);
      }
      if (row[prefix + 'selection_status'] !== template.selection_status) throw new Error(`Unexpected selection status for ${id}, ${target}.`);
      statuses[target][status] = (statuses[target][status] || 0) + 1;
      predictions[target] = {
        status, selection_status: row[prefix + 'selection_status'], predicted_label: label || null,
        confidence, raw_confidence: confidence, calibrated_confidence: score(row[prefix + 'calibrated_confidence'], id),
        probabilities: null, calibrated_probabilities: null, label_mapping: template.label_mapping,
        review_required: true, review_reasons: row[prefix + 'review_reasons'].split('; ').filter(Boolean),
        provenance: {model_id: modelId || null}, error: null,
      };
    }
    const source = Object.fromEntries(headers.filter(key => !key.startsWith('catalogiq:')).map(key => [key, row[key]]));
    const record = { product: {record_id: id, ...Object.fromEntries(FIELDS.map(field => [field, source[field] || '']))},
      source, result: {record_id: id, review_required: true, predictions}, mode: 'imported',
      engine_version: report.engine_version, relationship_registry_version: report.relationship_registry_version,
      decision: null, history: [], relationship_warnings: [] };
    byId.set(id, record);
    return record;
  });
  for (const target of TARGETS) {
    for (const status of ['suggestion', 'unavailable', 'failed']) {
      if ((statuses[target][status] || 0) !== (report.status_counts?.[target]?.[status] || 0)) {
        throw new Error(`The ${target} status counts do not match the report.`);
      }
    }
  }

  const warningsTable = readCSVTable(warningsText, {allowEmpty: true});
  requireColumns(warningsTable.headers, WARNING_COLUMNS, 'Relationship warnings CSV');
  // Check completeness against the same candidate pair inventory as flag_all.
  // This validates the supplied warnings and never changes a prediction or review decision.
  const expected = new Set();
  const relations = new Map();
  for (const relation of registry.relations.filter(relation => relation.status === 'candidate' && relation.allowed_pairs?.length)) {
    relations.set(relation.relation_id, relation);
    const allowed = new Set(relation.allowed_pairs.map(pair => JSON.stringify(pair)));
    for (const record of records) {
      const child = record.result.predictions[relation.child_target]?.predicted_label?.trim() || '';
      const parent = record.result.predictions[relation.parent_target]?.predicted_label?.trim() || '';
      let code = null;
      if (!child || !parent) code = 'missing_relationship_classification';
      else if (!allowed.has(JSON.stringify([child, parent]))) code = 'invalid_approved_classification_relationship';
      if (code) expected.add(JSON.stringify([record.product.record_id, relation.relation_id, code]));
    }
  }
  const seen = new Set();
  for (const row of warningsTable.records) {
    const record = byId.get(row.record_id);
    if (!record) throw new Error(`Warning references unknown product ${row.record_id}.`);
    const relation = relations.get(row.relation_id);
    if (!Object.hasOwn(RELATIONSHIP_MESSAGES, row.observation) || row.relation_status !== 'candidate' ||
        boolean(row.is_approved_decision, row.record_id) || row.registry_version !== report.relationship_registry_version ||
        !relation || row.child_target !== relation.child_target || row.parent_target !== relation.parent_target) {
      throw new Error(`Invalid candidate warning for ${row.record_id}.`);
    }
    for (const side of ['child', 'parent']) {
      const prediction = record.result.predictions[row[`${side}_target`]];
      if (!prediction || row[`${side}_prediction`] !== (prediction.predicted_label || '')) {
        throw new Error(`Warning labels do not match the saved prediction for ${row.record_id}.`);
      }
      const confidence = score(row[`${side}_confidence`], row.record_id);
      if ((confidence === null) !== (prediction.confidence === null) ||
          (confidence !== null && Math.abs(confidence - prediction.confidence) > 1e-9)) {
        throw new Error(`Warning confidence does not match the saved prediction for ${row.record_id}.`);
      }
    }
    const missing = !row.child_prediction || !row.parent_prediction;
    if (missing !== (row.observation === 'missing_relationship_classification')) throw new Error(`Warning code disagrees with its labels for ${row.record_id}.`);
    const key = JSON.stringify([row.record_id, row.relation_id, row.observation]);
    if (!expected.has(key)) throw new Error(`Warning disagrees with the candidate registry for ${row.record_id}.`);
    if (seen.has(key)) throw new Error(`Duplicate warning for ${row.record_id}.`);
    seen.add(key);
    record.relationship_warnings.push({...row, is_approved_decision: false});
  }
  if (seen.size !== expected.size) throw new Error(`The warnings file is incomplete. Found ${seen.size} warnings; expected ${expected.size} for these predictions.`);
  return {records, report, warningCount: warningsTable.records.length,
    flaggedCount: records.filter(record => record.relationship_warnings.length > 0).length};
}
