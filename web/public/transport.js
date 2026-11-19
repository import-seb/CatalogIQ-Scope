import {FIELDS,TARGETS,MAX_ROWS,MAX_BYTES,SCHEMA} from './domain.js';

// Shared browser-only demo. Uploaded product data never leaves the browser in this mode.
export function demoPredictions(products, examples) {
  if (!Array.isArray(products)||!products.length||products.length>MAX_ROWS) throw new Error(`Choose between 1 and ${MAX_ROWS} products.`);
  const ids=new Set();
  const output=products.map(row=>{
    if (!row||typeof row.record_id!=='string'||!row.record_id.trim()||ids.has(row.record_id)) throw new Error('Each product needs a unique, nonempty text ID.');
    ids.add(row.record_id);
    if (Object.keys(row).some(k=>!['record_id','quarantined',...FIELDS].includes(k))) throw new Error('Send only the supported product fields.');
    if (FIELDS.some(f=>!Object.hasOwn(row,f)||(row[f]!==null&&typeof row[f]!=='string'))) throw new Error('All five product fields must be text or null.');
    if (Object.hasOwn(row,'quarantined')&&typeof row.quarantined!=='boolean') throw new Error('quarantined must be true or false.');
    const index=examples.request.products.findIndex(example=>FIELDS.every(f=>row[f]===example[f])&&!row.quarantined);
    if (index!==-1){const saved=structuredClone(examples.response.products[index]);saved.record_id=row.record_id;return saved;}
    return {record_id:row.record_id,review_required:true,predictions:Object.fromEntries(TARGETS.map(target=>[target,{
      status:'unavailable',selection_status:target==='Platform'?'deferred':'selected',predicted_label:null,
      confidence:null,raw_confidence:null,probabilities:null,calibrated_probabilities:null,calibrated_confidence:null,
      label_mapping:{},review_required:true,review_reasons:[target==='Platform'?'target_unavailable':'models_not_connected'],
      lookup_status:null,provenance:{},error:null,
    }]))};
  });
  return {...structuredClone(examples.response),products:output};
}

export function createTransport({staticDemo=false,fetcher=globalThis.fetch}={}) {
  let examples;
  async function getJSON(path){const response=await fetcher(path);if(!response.ok)throw new Error('The workspace files could not be loaded. Refresh the page to try again.');return response.json();}
  return {
    async initialize(){
      if(staticDemo){examples=await getJSON('/examples.json');return {health:{mode:'demo',browser_only:true,max_rows:MAX_ROWS,max_bytes:MAX_BYTES},examples};}
      const [health,data]=await Promise.all([getJSON('/api/health'),getJSON('/api/examples')]);examples=data;return {health,examples};
    },
    async predict(products){
      if(staticDemo){if(!examples)throw new Error('The demo is still loading. Try again.');return demoPredictions(products,examples);}
      let response;
      try{response=await fetcher('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({schema_version:SCHEMA,products})});}
      catch{throw new Error('Couldn’t reach the local server. Keep this page open, restart the server, and try again.');}
      let data;try{data=await response.json();}catch{throw new Error('The server returned an unreadable response. No results were applied.');}
      if(!response.ok)throw new Error(data.error||'Classification is unavailable. Try again later.');return data;
    },
  };
}
