import fs from 'node:fs';
import {TARGETS, csvCell} from '../../public/domain.js';

export const response = JSON.parse(fs.readFileSync(new URL('../../../docs/integration/inference_adapter_response.json', import.meta.url)));
export const registry = JSON.parse(fs.readFileSync(new URL('../../../src/catalogiq/resources/relationships/classification_relationship_registry.json', import.meta.url)));
export const warningHeaders = ['record_id', 'relation_id', 'observation', 'child_target', 'child_prediction',
  'child_confidence', 'parent_target', 'parent_prediction', 'parent_confidence', 'relation_status',
  'registry_version', 'is_approved_decision'];
export function toCSV(rows, headers = Object.keys(rows[0])) {
  return [headers.map(csvCell).join(','), ...rows.map(row => headers.map(key => csvCell(row[key])).join(','))].join('\r\n');
}

// Synthetic source rows in Maria's documented dashboard_table format.
export function savedFiles({count = 3, conflicts = 1, missing = false} = {}) {
  const rows = [];
  const warnings = [];
  const statusCounts = Object.fromEntries(TARGETS.map(target => [target, {}]));
  for (let index = 0; index < count; index++) {
    const id = String(index + 1).padStart(6, '0');
    const row = {JoiningKey: id, Retailer: 'Synthetic retailer', Category: 'Synthetic', ProductCategory: 'Health',
      ProductBrand: 'Synthetic brand', ProductName: index === 1 ? '' : `Synthetic saved product ${index + 1}`,
      Sku: id, Upc: '', ProductUrl: '', 'catalogiq:mode': 'batch', 'catalogiq:review_status': '',
      'catalogiq:reviewer': '', 'catalogiq:reviewed_at': '', 'catalogiq:review_note': '',
      'catalogiq:engine_version': response.engine_version, 'catalogiq:registry_version': registry.registry_version,
      'catalogiq:eligibility': 'eligible', 'catalogiq:eligibility_version': 'target-eligibility-20261116-v2',
      'catalogiq:schema_version': response.schema_version, 'catalogiq:review_required': 'True'};
    for (const target of TARGETS) {
      const prediction = response.products[0].predictions[target];
      const prefix = `catalogiq:${target}:`;
      Object.assign(row, {[prefix + 'original_label']: prediction.predicted_label || '',
        [prefix + 'final_label']: '', [prefix + 'raw_confidence']: prediction.confidence ?? '',
        [prefix + 'calibrated_confidence']: prediction.calibrated_confidence ?? '',
        [prefix + 'prediction_status']: prediction.status, [prefix + 'selection_status']: prediction.selection_status,
        [prefix + 'review_required']: 'True', [prefix + 'review_reasons']: prediction.review_reasons.join('; '),
        [prefix + 'model_version']: prediction.provenance?.model_id || ''});
    }
    if (index < conflicts) row['catalogiq:Sub-Segment:original_label'] = 'Acid Relief';
    if (missing && index === 1) {
      row['catalogiq:Sub-Segment:original_label'] = '';
      row['catalogiq:Sub-Segment:raw_confidence'] = '';
      row['catalogiq:Sub-Segment:calibrated_confidence'] = '';
      row['catalogiq:Sub-Segment:prediction_status'] = 'unavailable';
    }
    for (const target of TARGETS) {
      const status = row[`catalogiq:${target}:prediction_status`];
      statusCounts[target][status] = (statusCounts[target][status] || 0) + 1;
    }
    rows.push(row);
    for (const relation of registry.relations.filter(relation => relation.status === 'candidate' && relation.allowed_pairs?.length)) {
      const child = row[`catalogiq:${relation.child_target}:original_label`];
      const parent = row[`catalogiq:${relation.parent_target}:original_label`];
      const allowed = relation.allowed_pairs.some(([a, b]) => a === child && b === parent);
      if (allowed) continue;
      warnings.push({record_id: id, relation_id: relation.relation_id,
        observation: !child || !parent ? 'missing_relationship_classification' : 'invalid_approved_classification_relationship',
        child_target: relation.child_target, child_prediction: child,
        child_confidence: row[`catalogiq:${relation.child_target}:raw_confidence`], parent_target: relation.parent_target,
        parent_prediction: parent, parent_confidence: row[`catalogiq:${relation.parent_target}:raw_confidence`],
        relation_status: 'candidate', registry_version: registry.registry_version, is_approved_decision: 'False'});
    }
  }
  const report = {eligibility_version: 'target-eligibility-20261116-v2', rows_in_file: count + 212,
    eligible: count, ineligible: 212, products_written: count, decisions_sha256: 'a'.repeat(64),
    schema_version: response.schema_version, engine_version: response.engine_version,
    relationship_registry_version: registry.registry_version, status_counts: statusCounts};
  return {rows, warnings, report, predictionsCSV: toCSV(rows), warningsCSV: toCSV(warnings, warningHeaders)};
}
