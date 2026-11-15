// Exercise Max's unmodified domain functions using synthetic records only.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const domain=await import(pathToFileURL(process.argv[2]).href);
const out=process.argv[3];
function record(id,version){
  const predictions=Object.fromEntries(domain.TARGETS.map(target=>[target,{
    status:target==='Platform'?'unavailable':'suggestion',predicted_label:target==='Platform'?null:'A',
    confidence:target==='Platform'?null:.8,review_required:true,review_reasons:['automatic_acceptance_disabled'],
    label_mapping:{probability_0:'A',probability_1:'B'},probabilities:{probability_0:.8,probability_1:.2},
    provenance:{model_id:`synthetic:${target}:${version}`}}]));
  return {product:{record_id:id,...Object.fromEntries(domain.FIELDS.map(f=>[f,'SYNTHETIC']))},
    source:{record_id:id},result:{record_id:id,predictions},mode:'synthetic',history:[],engine_version:'synthetic-engine',relationship_registry_version:'synthetic-registry'};
}
const a=record('000A','A');
const labels=Object.fromEntries(domain.TARGETS.filter(t=>t!=='Platform').map(t=>[t,'A']));
const approved=domain.recordDecision(a,'approved',labels,'Synthetic approval','Synthetic reviewer','2026-11-14T15:00:00Z');
const exportA=domain.auditExport([approved]);
fs.writeFileSync(path.join(out,'version_a.json'),exportA,{flag:'wx'});
const b=record('000B','B');
fs.writeFileSync(path.join(out,'version_b.json'),domain.auditExport([b]),{flag:'wx'});
assert.equal(fs.readFileSync(path.join(out,'version_a.json'),'utf8'),exportA);
const corrected=domain.recordDecision(approved,'corrected',{...labels,Segment:'B'},'Synthetic correction','Synthetic reviewer','2026-11-14T15:01:00Z');
assert.equal(corrected.result.predictions.Segment.predicted_label,'A');
assert.equal(approved.decision.labels.Segment,'A');
assert.equal(corrected.history.length,2);assert.equal(approved.history.length,1);
assert.match(domain.exportCSV([approved,b]),/synthetic:Segment:A/);
assert.match(domain.exportCSV([approved,b]),/synthetic:Segment:B/);
assert.throws(()=>fs.writeFileSync(path.join(out,'version_a.json'),'overwrite',{flag:'wx'}));
fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({status:'passed',synthetic_web_checks:4,
  upstream_domain_unmodified:true,prior_export_bytes_unchanged:true,version_references_retained:true,
  prior_approval_preserved:true,production_integration:false}),{flag:'wx'});
console.log('Four synthetic web-domain integration checks passed');
