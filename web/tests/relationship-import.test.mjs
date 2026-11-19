import test from 'node:test';
import assert from 'node:assert/strict';
import {parseSavedResults, warningMessage} from '../public/relationship-import.js';
import {TARGETS, recordDecision, exportCSV, auditExport} from '../public/domain.js';
import {savedFiles, response, registry, toCSV, warningHeaders} from './shared/saved-result-fixture.mjs';

function parse(files) {
  return parseSavedResults(files.predictionsCSV, files.warningsCSV, files.report, response, registry);
}

test('import aligns warning IDs, preserves scores and allows nameless eligible products', () => {
  const files = savedFiles({missing: true});
  files.warningsCSV = toCSV([...files.warnings].reverse(), warningHeaders);
  const parsed = parse(files);
  assert.equal(parsed.records.length, 3);
  assert.equal(parsed.warningCount, 2);
  assert.equal(parsed.flaggedCount, 2);
  assert.equal(parsed.records[0].product.record_id, '000001');
  assert.equal(parsed.records[1].product.ProductName, '');
  assert.equal(parsed.records[0].result.predictions.Segment.confidence, response.products[0].predictions.Segment.confidence);
  assert.equal(parsed.records[0].result.predictions.Segment.probabilities, null);
  assert.match(warningMessage(parsed.records[0].relationship_warnings[0]), /may not belong together/);
  assert.match(warningMessage(parsed.records[1].relationship_warnings[0]), /classification.*missing/);
});

test('candidate warning never blocks approval or correction and survives both exports', () => {
  const record = parse(savedFiles()).records[0];
  const original = structuredClone(record.result);
  const labels = Object.fromEntries(TARGETS.filter(t => t !== 'Platform').map(t => [t, record.result.predictions[t].predicted_label]));
  const approved = recordDecision(record, 'approved', labels, '', 'Reviewer');
  assert.equal(approved.decision.action, 'approved');
  const corrected = recordDecision(approved, 'corrected', {...labels, 'Sub-Segment': 'Oral Antihistamine'}, 'Checked the listing', 'Reviewer');
  assert.deepEqual(corrected.result, original);
  assert.equal(corrected.history.length, 2);
  assert.deepEqual(JSON.parse(auditExport([corrected])).records[0].relationship_warnings, record.relationship_warnings);
  assert.match(exportCSV([corrected]), /invalid_approved_classification_relationship/);
  assert.equal(record.relationship_warnings[0].relation_status, 'candidate');
});

test('rejects superseded eligibility, mismatched counts, versions and model vocabularies', () => {
  for (const change of [
    files => files.report.eligibility_version = 'old',
    files => files.report.products_written++,
    files => files.report.status_counts.Segment.suggestion++,
    files => files.report.decisions_sha256 = '',
    files => files.rows[0]['catalogiq:engine_version'] = 'other-run',
    files => files.rows[0]['catalogiq:Brand:model_version'] = 'other-model',
    files => files.rows[0]['catalogiq:Brand:original_label'] = 'not-a-model-class',
    files => files.rows[0]['catalogiq:Segment:raw_confidence'] = '',
    files => files.rows[0]['catalogiq:eligibility'] = 'quarantine',
    files => files.rows[0]['catalogiq:review_status'] = 'approved',
    files => files.rows[1].JoiningKey = files.rows[0].JoiningKey,
  ]) {
    const files = savedFiles();
    change(files);
    files.predictionsCSV = toCSV(files.rows);
    assert.throws(() => parse(files));
  }
});

test('rejects truncated, duplicated, alien, stale and falsely approved warning exports', () => {
  for (const change of [
    files => files.warnings.pop(),
    files => files.warnings.push({...files.warnings[0]}),
    files => files.warnings[0].record_id = '1',
    files => files.warnings[0].child_prediction = 'Cold/Flu',
    files => files.warnings[0].child_confidence = '0.01',
    files => files.warnings[0].registry_version = 'old',
    files => files.warnings[0].is_approved_decision = 'True',
    files => files.warnings[0].relation_status = 'approved',
    files => files.warnings[0].relation_id = 'unknown',
    files => files.warnings[0].observation = 'relationship_conflict',
    files => files.warnings[0].parent_target = 'Brand',
  ]) {
    const files = savedFiles();
    change(files);
    files.warningsCSV = toCSV(files.warnings, warningHeaders);
    assert.throws(() => parse(files));
  }
});

test('accepts an empty warning file only when every candidate pair is valid', () => {
  const result = parse(savedFiles({conflicts: 0}));
  assert.equal(result.warningCount, 0);
  assert.equal(result.flaggedCount, 0);
});

test('imports the full run size and reconciles 651 synthetic warnings before selecting UI batches', () => {
  const result = parse(savedFiles({count: 27870, conflicts: 651}));
  assert.equal(result.records.length, 27870);
  assert.equal(result.warningCount, 651);
  assert.equal(result.flaggedCount, 651);
  assert.equal(result.records.at(-1).product.record_id, '027870');
});
