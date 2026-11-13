import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {parseCSV,alignResponse,recordDecision,exportCSV,auditExport,FIELDS,TARGETS,alternatives,canReview} from '../public/domain.js';
const request=JSON.parse(fs.readFileSync(new URL('../../docs/integration/inference_adapter_request.json',import.meta.url)));
const response=JSON.parse(fs.readFileSync(new URL('../../docs/integration/inference_adapter_response.json',import.meta.url)));
const header=['record_id',...FIELDS];
const row='0001,Health,Name,Brand,Description,Contents';
const base=()=>({product:request.products[0],source:{...request.products[0],vendor_note:'=SUM(A1)'},result:structuredClone(response.products[0]),mode:'demo',history:[]});
const labels=r=>Object.fromEntries(TARGETS.filter(t=>r.result.predictions[t].status==='suggestion').map(t=>[t,r.result.predictions[t].predicted_label]));
test('CSV preserves IDs, source labels and embedded quoted text but sends only allowed features',()=>{
  const csv='\uFEFF'+header.join(',')+',Segment,Note\r\n0001,Health,"Name, with comma",Brand,"Line 1\nLine 2",,"Existing label","He said ""yes"""\r\n';
  const [r]=parseCSV(csv);assert.equal(r.product.record_id,'0001');assert.equal(r.product.ProductName,'Name, with comma');assert.equal(r.product.ProductDescription,'Line 1\nLine 2');assert.equal(r.source.Segment,'Existing label');assert.equal(r.source.Note,'He said "yes"');assert.equal(r.product.Segment,undefined);
});
test('CSV rejects missing and duplicate IDs, headers, malformed quotes and ragged rows',()=>{
  for(const csv of [header.join(',')+'\n'+row+'\n'+row,'record_id,record_id\nx,y',header.join(',')+'\n,Health,Name,Brand,Description,Contents',header.join(',')+'\n0001,Health,"bad',header.join(',')+'\n0001,Health,"bad"x,Brand,Description,Contents',header.join(',')+'\n0001,x'])assert.throws(()=>parseCSV(csv));
});
test('Quarantine is preserved and invalid values rejected',()=>{assert.equal(parseCSV(header.join(',')+',quarantined\n'+row+',true')[0].product.quarantined,true);assert.throws(()=>parseCSV(header.join(',')+',quarantined\n'+row+',yes'));});
test('Response alignment is based on exact ID, never positional order',()=>{const shuffled={...response,products:[...response.products].reverse()};assert.deepEqual(alignResponse(shuffled,request.products).map(r=>r.record_id),request.products.map(r=>r.record_id));});
test('Duplicate, missing, alien IDs and invalid scores are rejected atomically',()=>{
  for(const mutate of [r=>r.products.pop(),r=>r.products[1].record_id=r.products[0].record_id,r=>r.products[0].record_id='alien',r=>r.products[0].predictions.Segment.confidence=1.5]){const copy=structuredClone(response);mutate(copy);assert.throws(()=>alignResponse(copy,request.products));}
});
test('Top alternatives use explicit mapping and numerical probabilities',()=>{const p={label_mapping:{probability_10:'C',probability_2:'B',probability_0:'A'},probabilities:{probability_10:.8,probability_2:.15,probability_0:.05}};assert.equal(alternatives(p)[0].label,'C');});
test('Review requires reviewer, all supported targets and an available prediction',()=>{const r=base();assert.ok(canReview(r));assert.throws(()=>recordDecision(r,'approved',labels(r),'',''));assert.throws(()=>recordDecision(r,'approved',{},'','Max'));r.result.predictions.Brand.status='unavailable';assert.throws(()=>recordDecision(r,'approved',labels(r),'','Max'));});
test('Corrections preserve model output and append an attributable history',()=>{
  const r=base();const changed=labels(r);changed.Segment=Object.values(r.result.predictions.Segment.label_mapping).find(l=>l!==changed.Segment);
  assert.throws(()=>recordDecision(r,'corrected',changed,'','Max'));
  const saved=recordDecision(r,'corrected',changed,'Checked listing','Max','2026-11-12T12:00:00Z');
  assert.equal(saved.result.predictions.Segment.predicted_label,r.result.predictions.Segment.predicted_label);assert.equal(saved.decision.labels.Segment,changed.Segment);assert.equal(saved.history.length,1);assert.equal(r.history.length,0);
  const rejected=recordDecision(saved,'rejected',changed,'Wrong listing','Max');assert.equal(rejected.history.length,2);assert.deepEqual(rejected.decision.labels,{});
});
test('CSV separates raw and final labels, escapes formulas; JSON retains exact originals',()=>{
  const r=base();const csv=exportCSV([r]);assert.match(csv,/"'=SUM\(A1\)"/);assert.match(csv,/catalogiq:mode/);assert.equal(JSON.parse(auditExport([r])).records[0].source.vendor_note,'=SUM(A1)');
  const grid=csv.replace(/^\uFEFF/,'').split('\r\n').map(line=>line.match(/"(?:[^"]|"")*"/g).map(cell=>cell.slice(1,-1).replaceAll('""','"')));
  assert.equal(grid[1][grid[0].indexOf('catalogiq:Segment:final_label')],'');assert.equal(grid[1][grid[0].indexOf('catalogiq:Segment:original_label')],'Allergy');
});
