import {FIELDS,TARGETS,TITLES,MAX_ROWS,MAX_BYTES,SCHEMA,parseCSV,alignResponse,alternatives,canReview,recordDecision,exportCSV,auditExport,csvCell} from './domain.js';
import {createTransport} from './transport.js';

const transport=createTransport({staticDemo:document.querySelector('meta[name="catalogiq-runtime"]')?.content==='static-demo'});

const $ = s => document.querySelector(s);
const esc = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const paths = {
  grid:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
  check:'m5 12 4 4 10-10', chevron:'m9 5 7 7-7 7', close:'m6 6 12 12 M18 6 6 18',
  review:'M9 3h6l1 3h4v15H4V6h4l1-3z M9 12h6 M9 16h4',
  box:'m3 7 9-4 9 4v11l-9 4-9-4V7z m0 0 9 4 9-4 M12 11v11',
  upload:'M12 16V3 m-5 5 5-5 5 5 M4 15v6h16v-6',
  download:'M12 3v13 m-5-5 5 5 5-5 M4 17v4h16v-4', plus:'M12 5v14 M5 12h14',
  search:'M21 21l-5-5 M18 10a8 8 0 1 0-16 0 8 8 0 0 0 16 0',
  info:'M12 11v6 M12 7v.2 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  edit:'m14 4 6 6 M4 20l1-6L17 2l5 5-12 12-6 1z',
  clock:'M12 7v5l3 2 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  file:'M14 2H4v20h16V8l-6-6z M14 2v6h6 M8 13h8 M8 17h6',
  refresh:'M20 7v5h-5 M4 17v-5h5 M5 8a8 8 0 0 1 13-3l2 2 M19 16a8 8 0 0 1-13 3l-2-2',
  shield:'m12 2 8 4v6c0 6-8 10-8 10S4 18 4 12V6l8-4z m-4 10 3 3 5-6',
};
const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.box}"/></svg>`;
const KEY='catalogiq-workspace-v1';
let state={version:1,records:[],selectedId:null,filter:'all',search:'',reviewer:''};
let health={mode:'demo'}, examples, busy=false, editing=false, storageOK=true, staged=[], stagedName='', uploadToken=0;
const status = r => r.decision?.action || (canReview(r) ? 'pending' : 'waiting');
const statusLabel = value => ({pending:'Needs review',waiting:'Awaiting models',approved:'Approved',corrected:'Corrected',rejected:'Rejected'}[value]);
const percent = p => Number.isFinite(p) ? (p<1&&p>=0.9995?'>99.9%':`${(100*p).toFixed(1)}%`) : '—';
function notify(message){$('#toast').textContent=message;$('#toast').classList.add('show');clearTimeout(notify.timer);notify.timer=setTimeout(()=>$('#toast').classList.remove('show'),4500);}
function save(){try{localStorage.setItem(KEY,JSON.stringify(state));storageOK=true;}catch{storageOK=false;notify('Browser storage is full or unavailable. Export your work before closing.');}}
function selected(){return state.records.find(r=>r.product.record_id===state.selectedId);}
function counts(){return {all:state.records.length,pending:state.records.filter(r=>!r.decision).length,reviewed:state.records.filter(r=>r.decision&&r.decision.action!=='rejected').length,rejected:state.records.filter(r=>r.decision?.action==='rejected').length};}
function visibleRecords(){return state.records.filter(r=> (state.filter==='all'||(state.filter==='pending'&&!r.decision)||(state.filter==='reviewed'&&r.decision&&r.decision.action!=='rejected')||(state.filter==='rejected'&&r.decision?.action==='rejected'))&&[r.product.record_id,...FIELDS.map(f=>r.product[f])].join(' ').toLowerCase().includes(state.search.toLowerCase()));}
const pill = r => `<span class="pill ${status(r)}">${status(r)==='approved'||status(r)==='corrected'?'✓ ':''}${statusLabel(status(r))}</span>`;
function sourceDetails(r){return `<details class="source-summary"><summary>Product details</summary><dl>${FIELDS.map(f=>`<dt>${esc(({ProductCategory:'Category',ProductName:'Name',ProductBrand:'Brand',ProductDescription:'Description',ProductContents:'Contents'})[f])}</dt><dd>${esc(r.product[f] || 'Not provided')}</dd>`).join('')}</dl></details>`;}
const reasonNames={automatic_acceptance_disabled:'Automatic approval is turned off.',no_approved_threshold:'Review thresholds have not been approved.',provisional_model:'Sub-segment suggestions are still provisional.',target_unavailable:'Platform classification is not available yet.',models_not_connected:'Models are not connected. No prediction was generated.',missing_text:'Some product information is missing.',quarantined:'This product was flagged before classification.',unapproved_relationship_rule:'This relationship has not been approved.',relationship_conflict:'A combination of predictions needs a closer look.'};
function reasonText(code){return reasonNames[code] || code.replaceAll('_',' ').replace(/^./,s=>s.toUpperCase())+'.';}

function render(){
  const c=counts();
  $('#app').innerHTML=`<div class="shell">
    <aside class="sidebar"><div class="brand"><img src="/assets/mark.svg" alt=""><div class="wordmark">Catalog<span>IQ</span></div></div>
      <div class="nav-caption">Workspace</div><nav aria-label="Main navigation">
      ${[['all','grid','Products',c.all],['pending','review','Review queue',c.pending],['reviewed','check','Reviewed',c.reviewed]].map(([f,i,label,n])=>`<button class="nav-item ${state.filter===f?'active':''}" data-action="filter" data-filter="${f}" aria-label="${label}" ${state.filter===f?'aria-current="page"':''}>${icon(i)}<span class="nav-text">${label}</span><span class="nav-number">${n}</span></button>`).join('')}
      </nav><div class="sidebar-bottom"><button class="nav-item" data-action="help" aria-label="Workspace guide">${icon('info')}<span class="nav-text">Workspace guide</span></button><div class="sidebar-note">Built with<br><img class="partner-logo" src="/assets/giggso.png" alt="Giggso"><br>Market Pulse Project</div></div></aside>
    <div class="main-shell"><header class="topbar"><div class="breadcrumb">Catalog management ${icon('chevron')} <strong>Product workspace</strong></div><div class="topbar-right"><span class="connection ${health.mode==='live'?'live':''}">${health.mode==='live'?'Models connected':'Demo workspace'}</span><span class="avatar" aria-label="Local workspace">MP</span></div></header>
    <main id="main" class="workspace"><div class="page-heading"><div><div class="eyebrow">A little clarity for every product</div><h1>Your product workspace</h1><p class="subline">Review classifications before they reach your catalog.</p></div><div class="actions"><button class="button" data-action="export" ${c.all?'':'disabled'}>${icon('download')}Export results</button><button class="button primary" data-action="add">${icon('plus')}Add products</button></div></div>
    ${health.mode==='demo'?`<div class="demo-notice">${icon('info')}<div><strong>You’re exploring a demo.</strong> The two example products have saved predictions. New products will wait for the models to be connected.</div><button data-action="help">How it works</button></div>`:''}
    <div class="workflow" aria-label="Workflow"><span><b>1</b>Add products</span><span class="connector" aria-hidden="true"></span><span class="current"><b>2</b>Review suggestions</span><span class="connector" aria-hidden="true"></span><span><b>3</b>Export results</span></div>
    <div class="work-grid"><div><section class="panel" aria-label="Products"><div class="panel-top"><h2>Products <span>${c.all} total</span></h2><span class="saved">${icon(storageOK?'check':'info')}${storageOK?'Saved in this browser':'Not saved — export your work'}</span></div>
    <div class="search-wrap">${icon('search')}<input id="search" type="search" aria-label="Search products" placeholder="Search products or IDs…" value="${esc(state.search)}"></div>
    <div class="filters" aria-label="Filter products">${[['all','All products',c.all],['pending','Needs review',c.pending],['reviewed','Reviewed',c.reviewed],['rejected','Rejected',c.rejected]].map(([f,l,n])=>`<button class="filter ${state.filter===f?'active':''}" data-action="filter" data-filter="${f}" aria-pressed="${state.filter===f}">${l}<span>${n}</span></button>`).join('')}</div>
    <div id="product-table">${tableHTML()}</div><div class="panel-bottom"><span>${visibleRecords().length} of ${c.all} products</span><button class="button quiet small" data-action="examples" ${busy?'disabled':''}>${icon('refresh')}Load examples</button></div></section>
    <p class="under-table"><strong>Every prediction gets a human check.</strong><br>Confidence is the model’s score, not a guarantee of correctness.</p>
    ${state.records.some(r=>!r.decision&&!canReview(r))?`<button class="button" data-action="classify" ${busy?'disabled':''}>${icon('refresh')}${busy?'Checking products…':'Try classification'}</button>`:''}
    </div><section class="panel review-panel" id="review-panel" aria-label="Product review">${reviewHTML()}</section></div>
    <footer class="footer-note"><span>${health.browser_only?'Uploads and reviews stay in your browser. They aren’t shared with other visitors.':health.mode==='demo'?'Demo results are marked in every export.':'Product reviews are stored in this browser.'} Your original product data is preserved.</span><button class="button quiet small" data-action="reset">Clear workspace</button></footer></main></div></div>`;
}

function tableHTML(){
  const records=visibleRecords();
  if(!records.length)return `<div class="empty">${icon('box')}<h3>${state.records.length?'No products match this view':'Your next catalog starts here'}</h3><p>${state.records.length?'Try another filter or search term.':'Add a product, upload a CSV, or explore the two examples.'}</p><div class="actions"><button class="button primary" data-action="${state.records.length?'clear-filter':'add'}">${state.records.length?'Show all products':'Add products'}</button></div></div>`;
  return `<div class="table-scroll"><table><thead><tr><th scope="col">PRODUCT</th><th scope="col" class="table-segment">SEGMENT</th><th scope="col">CONFIDENCE</th><th scope="col">REVIEW</th></tr></thead><tbody>${records.map((r,i)=>{
    const p=r.result?.predictions.Segment;
    return `<tr class="${r.product.record_id===state.selectedId?'selected':''}"><td><div class="product-cell"><div class="product-art ${i%2?'purple':''}">${esc((r.product.ProductBrand||r.product.ProductName||'P').slice(0,2).toUpperCase())}</div><button class="product-name" data-action="select" data-id="${esc(r.product.record_id)}">${esc(r.product.ProductName || 'Unnamed product')}<span class="product-id">${esc(r.product.record_id)}</span></button></div></td><td class="table-segment">${esc(p?.predicted_label||'Not available')}</td><td><span class="confidence-number">${percent(p?.confidence)}</span>${Number.isFinite(p?.confidence)?`<progress class="score-progress" value="${p.confidence}" max="1" aria-label="Segment confidence"></progress>`:''}</td><td>${pill(r)}</td></tr>`;
  }).join('')}</tbody></table></div>`;
}

function reviewHTML(){
  const r=selected();
  if(!r)return `<div class="empty">${icon('review')}<h3>A closer look, one product at a time</h3><p>Select a product to see its predictions and review details.</p></div>`;
  const predictions=r.result?.predictions||{};
  const reasons=[...new Set(Object.values(predictions).flatMap(p=>p.review_reasons||[]))].filter(c=>c!=='target_unavailable');
  const reviewable=canReview(r);
  const anySuggestion=Object.values(predictions).some(p=>p.status==='suggestion');
  const failedTargets=Object.values(predictions).some(p=>p.status==='failed');
  const modelSource=r.mode==='demo'?'Saved example · not a new model run':'Model suggestions';
  return `<div class="review-header"><h2>Product review</h2><small>${state.records.indexOf(r)+1} of ${state.records.length}</small></div><div class="review-body">
    <div class="review-product"><div class="product-art">${icon('box')}</div><div><h3>${esc(r.product.ProductName||'Unnamed product')}</h3><p>${esc(r.product.record_id)} · ${esc(modelSource)}</p></div></div>
    ${sourceDetails(r)}<div class="review-callout">${icon('shield')}<div>${reviewable?'<strong>Your review is required.</strong> '+(reasons.slice(0,2).map(reasonText).join(' ')||'Check the suggestions before approving.'):anySuggestion||failedTargets?'<strong>Some predictions are unavailable.</strong> Your product is saved. Try classification again before approving.':'<strong>Waiting for models.</strong> Your product is saved. Predictions will be available after the models are connected.'}</div></div>
    <div class="section-label">${r.decision&&r.decision.action!=='rejected'?'Reviewed classifications':'Suggested classifications'}<span>${editing?'Choose a label below':'Raw confidence'}</span></div>
    <div class="prediction-grid">${TARGETS.map(t=>{
      const p=predictions[t];const usable=p?.status==='suggestion';const final=r.decision?.labels[t]??p?.predicted_label;
      return `<div class="prediction ${usable?'':'unavailable'}"><div class="prediction-name"><span>${TITLES[t]}</span><strong>${usable?percent(p.confidence):'—'}</strong></div>
      ${editing&&usable?`<label class="visually-hidden" for="label-${t}">${TITLES[t]}</label><select id="label-${t}" data-target="${t}" aria-label="Correct ${TITLES[t]}">${Object.values(p.label_mapping).map(label=>`<option value="${esc(label)}" ${label===final?'selected':''}>${esc(label)}</option>`).join('')}</select>`:`<div class="prediction-value">${esc(usable?final:t==='Platform'?'Not available':p?.status==='failed'?'Couldn’t classify':'Awaiting models')}</div>`}
      ${p?.selection_status==='provisional'?'<p class="micro">Provisional suggestion</p>':''}${r.decision?.labels[t]&&r.decision.labels[t]!==p?.predicted_label?`<p class="micro">Changed from ${esc(p.predicted_label)}. Score belongs to original prediction.</p>`:''}</div>`;
    }).join('')}</div>
    ${reviewable?`<details class="alternative-details"><summary>See alternatives & review notes</summary><div class="alternatives">${TARGETS.filter(t=>predictions[t]?.status==='suggestion').map(t=>`<div class="alternative-target">${TITLES[t]}</div>${alternatives(predictions[t]).map(a=>`<div class="alternative-line"><b>${esc(a.label)}</b><span>${percent(a.probability)}</span></div>`).join('')}`).join('')}</div><p class="micro">${esc(reasons.map(reasonText).join(' '))}</p><p class="micro">Alternatives are suggestions from the model, not approved relationships.</p></details>`:''}
    ${r.decision&&!editing?`<div class="decision-banner ${r.decision.action}"><strong>${statusLabel(r.decision.action)}</strong> by ${esc(r.decision.reviewer)}<br><span>${esc(new Date(r.decision.timestamp).toLocaleString())}</span>${r.decision.note?`<p>${esc(r.decision.note)}</p>`:''}<button class="button small" data-action="edit">${icon('edit')}Review again</button></div>`:`<div class="review-controls"><div class="field"><label class="field-label" for="reviewer">Reviewed by</label><input class="text-input" id="reviewer" autocomplete="name" placeholder="Your name" value="${esc(state.reviewer)}" maxlength="100"><p class="micro">Recorded with your decision. This is a local workspace, not a verified login.</p></div><label class="field-label" for="review-note">Review note <span class="optional">${editing?'· required for a correction':'· optional for approval'}</span></label><textarea id="review-note" placeholder="Explain a correction or why you’re rejecting this product…" maxlength="2000">${esc(editing?r.decision?.note||'':'')}</textarea><p id="review-error" class="error-text" role="alert" hidden></p><div class="review-actions"><button class="button primary" data-action="${editing?'correct':'approve'}" ${reviewable?'':'disabled'}>${icon('check')}${editing?'Save review':'Approve'}</button><button class="button" data-action="${editing?'cancel-edit':'edit'}" ${reviewable?'':'disabled'}>${icon(editing?'close':'edit')}${editing?'Cancel':'Correct'}</button><button class="button danger" data-action="reject">Reject</button></div><p class="review-hint">${reviewable?'Approval covers the five available targets. Platform stays unavailable. Your decision does not change the model’s review policy.':'Approval and corrections unlock when predictions are available.'}</p></div>`}
    ${r.history?.length?`<details class="alternative-details"><summary>Review history (${r.history.length})</summary><ol class="history-list">${r.history.map(h=>`<li>${esc(statusLabel(h.action))} · ${esc(h.reviewer)} · ${esc(new Date(h.timestamp).toLocaleString())}${h.note?`<br>${esc(h.note)}`:''}</li>`).join('')}</ol></details>`:''}</div>`;
}

function newRecords(inputs, response, mode, sourceName){
  const aligned=alignResponse(response,inputs.map(x=>x.product));
  return inputs.map((x,i)=>({...x,result:aligned[i],mode,source_name:sourceName,created_at:new Date().toISOString(),engine_version:response.engine_version,relationship_registry_version:response.relationship_registry_version,decision:null,history:[]}));
}
function addExamples(){
  const existing=new Set(state.records.map(r=>r.product.record_id));
  const inputs=examples.request.products.map(product=>({product:structuredClone(product),source:structuredClone(product)}));
  const added=newRecords(inputs,examples.response,'demo','Supplied examples').filter(r=>!existing.has(r.product.record_id));
  if(state.records.length+added.length>MAX_ROWS){notify(`Keep this workspace to ${MAX_ROWS} products.`);return;}
  state.records.push(...added);state.selectedId=added[0]?.product.record_id||state.records[0]?.product.record_id;state.filter='all';state.search='';editing=false;save();render();notify(added.length?'Example products loaded.':'The examples are already in your workspace.');
}
async function requestPredictions(products){
  return transport.predict(products);
}
async function importProducts(inputs,name){
  if(state.records.length+inputs.length>MAX_ROWS)throw new Error(`This workspace supports ${MAX_ROWS} products. Export and clear it before adding another batch.`);
  const existing=new Set(state.records.map(r=>r.product.record_id));
  const duplicate=inputs.find(r=>existing.has(r.product.record_id));
  if(duplicate)throw new Error(`Product ID “${duplicate.product.record_id}” already exists. Use a different ID, or clear the current workspace.`);
  const data=await requestPredictions(inputs.map(x=>x.product));
  const added=newRecords(inputs,data,health.mode,name);
  state.records.push(...added);state.selectedId=added[0].product.record_id;state.filter='all';state.search='';editing=false;save();
}
function dialogHeader(id,title,subtitle){return `<div class="dialog-header"><div><h2 id="${id}">${title}</h2><p>${subtitle}</p></div><button class="icon-button" data-close aria-label="Close dialog">${icon('close')}</button></div>`;}
function showInput(tab='csv'){
  uploadToken++;staged=[];stagedName='';
  const dialog=$('#input-dialog');
  dialog.innerHTML=dialogHeader('input-title','Add products','Start with one product or bring in a CSV from your store.')+`<div class="dialog-content"><div class="dialog-tabs" aria-label="Input method"><button class="dialog-tab ${tab==='csv'?'active':''}" data-tab="csv" aria-pressed="${tab==='csv'}">Upload CSV</button><button class="dialog-tab ${tab==='single'?'active':''}" data-tab="single" aria-pressed="${tab==='single'}">One product</button></div>
    ${tab==='csv'?`<label class="dropzone" id="dropzone">${icon('upload')}<strong>Drop your CSV here, or browse</strong><p>Up to ${MAX_ROWS} products · CSV · 2 MB maximum</p><input type="file" id="csv-file" accept=".csv,text/csv" aria-label="Upload product CSV"></label><div id="file-ready" class="file-ready" hidden></div><button class="button small" id="template">${icon('download')}Download CSV template</button><p class="micro">Keep product IDs as text to preserve leading zeros. Extra columns are retained for export but are not sent to the model.</p>`:`<form id="single-form"><div class="form-grid"><div class="field"><label class="field-label" for="record-id">Product ID</label><input id="record-id" class="text-input" name="record_id" placeholder="e.g. SKU-001" required maxlength="200"></div><div class="field"><label class="field-label" for="product-brand">Brand</label><input id="product-brand" class="text-input" name="ProductBrand" placeholder="e.g. Claritin" maxlength="1000"></div><div class="field full"><label class="field-label" for="product-name">Product name</label><input id="product-name" class="text-input" name="ProductName" placeholder="e.g. Claritin RediTabs" required maxlength="2000"></div><div class="field full"><label class="field-label" for="product-category">Store category</label><input id="product-category" class="text-input" name="ProductCategory" placeholder="e.g. Health > Allergy Relief" maxlength="4000"><p>Use the category from the product’s original listing.</p></div><div class="field full"><label class="field-label" for="product-description">Description</label><textarea id="product-description" name="ProductDescription" placeholder="What does the listing say?" maxlength="20000"></textarea></div><div class="field full"><label class="field-label" for="product-contents">Contents or ingredients</label><input id="product-contents" class="text-input" name="ProductContents" placeholder="Optional" maxlength="10000"></div></div></form>`}
    <p class="micro">${health.mode==='demo'?'Demo mode: new products are saved with “Awaiting models.” No new predictions will be generated.':'Your products will be sent to the local classification adapter.'}</p><p class="error-text" id="input-error" role="alert" hidden></p><div class="dialog-footer"><button class="button quiet" data-close>Cancel</button><button class="button primary" id="import-submit" ${tab==='csv'?'disabled':''}>${icon('plus')}Add products</button></div></div>`;
  if(!dialog.open)dialog.showModal();
  dialog.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showInput(b.dataset.tab));
  if(tab==='csv'){
    $('#template').onclick=()=>download('catalogiq-product-template.csv',['record_id',...FIELDS].map(csvCell).join(',')+'\r\n'+['SKU-001','Health > Allergy Relief','Your product name','Your brand','Product description',''].map(csvCell).join(','),'text/csv;charset=utf-8');
    $('#csv-file').onchange=()=>readFile($('#csv-file').files[0]);
    $('#dropzone').ondragover=e=>{e.preventDefault();$('#dropzone').classList.add('drag');};
    $('#dropzone').ondragleave=()=>$('#dropzone').classList.remove('drag');
    $('#dropzone').ondrop=e=>{e.preventDefault();$('#dropzone').classList.remove('drag');readFile(e.dataTransfer.files[0]);};
  }else $('#single-form').onsubmit=e=>{e.preventDefault();$('#import-submit').click();};
  $('#import-submit').onclick=async()=>{
    if(busy)return;
    let inputs=staged,name=stagedName;
    if(tab==='single'){
      if(!$('#single-form').reportValidity())return;
      const data=Object.fromEntries(new FormData($('#single-form')));inputs=[{product:data,source:{...data}}];name='Single product entry';
    }
    if(!inputs.length)return;
    busy=true;dialog.querySelectorAll('button,input,textarea').forEach(el=>el.disabled=true);$('#import-submit').textContent='Adding products…';
    try{await importProducts(inputs,name);dialog.close();render();notify(`${inputs.length} product${inputs.length===1?'':'s'} added.`);}
    catch(e){showError('#input-error',e.message);}
    finally{busy=false;if(dialog.open){dialog.querySelectorAll('button,input,textarea').forEach(el=>el.disabled=false);$('#import-submit').textContent='Add products';}else render();}
  };
}
async function readFile(file){
  if(!file||busy)return;const token=++uploadToken;staged=[];$('#import-submit').disabled=true;$('#file-ready').hidden=true;$('#input-error').hidden=true;
  try{
    if(!file.name.toLowerCase().endsWith('.csv'))throw new Error('Choose a CSV file.');
    if(file.size>MAX_BYTES)throw new Error('Choose a file smaller than 2 MB.');
    const text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());
    if(token!==uploadToken||!$('#csv-file'))return;
    if(text.includes('\0'))throw new Error('Save the file as a UTF-8 CSV and try again.');
    staged=parseCSV(text);stagedName=file.name;$('#file-ready').textContent=`${file.name} · ${staged.length} products ready. Original IDs and extra columns will be preserved.`;$('#file-ready').hidden=false;$('#import-submit').disabled=false;
  }catch(e){if(token===uploadToken&&$('#csv-file'))showError('#input-error',e.message);}
}
function showError(selector,message){$(selector).textContent=message;$(selector).hidden=false;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function showExport(){
  const d=$('#export-dialog');d.innerHTML=dialogHeader('export-title','Export your workspace','Take your classifications and review decisions with you.')+`<div class="dialog-content"><label class="export-choice"><input type="radio" name="export-kind" value="csv" checked>Results spreadsheet (.csv)<span>Original source columns, model suggestions, confidence and your final decisions. Pending or rejected products have no final labels.</span></label><label class="export-choice"><input type="radio" name="export-kind" value="json">Full review record (.json)<span>Exact source values, all probabilities, model versions and the complete review history.</span></label><label class="inline-check"><input type="checkbox" id="reviewed-only">Only approved and corrected products</label><p class="micro">Demo rows are marked. CSV cells that could run spreadsheet formulas are escaped; the JSON file retains the exact originals.</p><p class="error-text" id="export-error" role="alert" hidden></p><div class="dialog-footer"><button class="button quiet" data-close>Cancel</button><button class="button primary" id="download-export">${icon('download')}Download export</button></div></div>`;d.showModal();
  $('#download-export').onclick=()=>{const rows=state.records.filter(r=>!$('#reviewed-only').checked||(r.decision&&r.decision.action!=='rejected'));if(!rows.length){showError('#export-error','There are no approved or corrected products to export yet.');return;}const kind=d.querySelector('[name="export-kind"]:checked').value;download(`catalogiq-${health.mode==='demo'?'demo-':''}results-${new Date().toISOString().slice(0,10)}.${kind}`,kind==='csv'?exportCSV(rows):auditExport(rows),kind==='csv'?'text/csv;charset=utf-8':'application/json');d.close();notify(`${rows.length} products exported.`);};
}
function showHelp(){const d=$('#help-dialog');d.innerHTML=dialogHeader('help-title','A human check for every product','A simple path from product information to reviewed classifications.')+`<div class="dialog-content help-copy"><h3>1. Add your products</h3><p>Enter one product or upload a UTF-8 CSV using the template. Every product needs a unique ID. Source data stays separate from model suggestions.</p><h3>2. Review the suggestions</h3><p>Check Segment, Sub-segment, Brand, Manufacturer and Age group. Platform is unavailable. Use Correct to choose another label, or Reject if the product should not be accepted. A correction or rejection needs a reason.</p><h3>What confidence means</h3><p>The percentage is the model’s raw score for its original suggestion. It does not guarantee accuracy. A correction does not inherit that score. Automatic approval is disabled, and relationship rules are awaiting approval.</p><h3>3. Keep a record</h3><p>Export a CSV for your catalog or JSON for the complete history. Reviews are saved only in this browser, not in a shared team account. A model suggestion remains separate from your final decision.</p><h3>About this demo</h3><p>The Claritin and Culturelle examples use synthetic descriptions and saved outputs supplied with the adapter. They are examples, not verified labels. Changed or new products do not receive sample predictions. Real inference requires the trained model files.</p></div>`;d.showModal();}
function decide(action){
  const r=selected();const labels={};
  for(const t of TARGETS)if(r.result?.predictions[t]?.status==='suggestion')labels[t]=editing?$(`[data-target="${t}"]`).value:r.result.predictions[t].predicted_label;
  state.reviewer=$('#reviewer').value;
  if(action==='corrected'&&!Object.keys(labels).some(t=>labels[t]!==r.result.predictions[t].predicted_label))action='approved';
  try{const next=recordDecision(r,action,labels,$('#review-note').value,state.reviewer);state.records[state.records.indexOf(r)]=next;editing=false;save();render();notify(`Review saved: ${statusLabel(action).toLowerCase()}.`);}
  catch(e){showError('#review-error',e.message);}
}

document.addEventListener('click',async e=>{
  const close=e.target.closest('[data-close]');if(close){if(!busy)close.closest('dialog').close();return;}
  const b=e.target.closest('[data-action]');if(!b||b.disabled)return;
  if(busy)return;
  const action=b.dataset.action;
  if(action==='select'){state.selectedId=b.dataset.id;editing=false;save();render();if(innerWidth<761)$('#review-panel').scrollIntoView({behavior:'smooth',block:'start'});}
  if(action==='filter'){state.filter=b.dataset.filter;editing=false;const rows=visibleRecords();if(!rows.some(r=>r.product.record_id===state.selectedId))state.selectedId=rows[0]?.product.record_id||null;save();render();}
  if(action==='clear-filter'){state.filter='all';state.search='';render();}
  if(action==='add')showInput();
  if(action==='examples')addExamples();
  if(action==='help')showHelp();
  if(action==='export')showExport();
  if(action==='edit'){editing=true;$('#review-panel').innerHTML=reviewHTML();}
  if(action==='cancel-edit'){editing=false;$('#review-panel').innerHTML=reviewHTML();}
  if(action==='approve')decide('approved');
  if(action==='correct')decide('corrected');
  if(action==='reject')decide('rejected');
  if(action==='reset'){
    const d=$('#reset-dialog');d.innerHTML=dialogHeader('reset-title','Clear this workspace?','Export your work first if you want to keep it.')+`<div class="dialog-content"><p class="help-copy">This removes ${state.records.length} products and their review history from this browser. Downloaded exports will remain.</p><div class="dialog-footer"><button class="button" data-close>Keep products</button><button class="button danger" id="confirm-reset">Clear workspace</button></div></div>`;d.showModal();$('#confirm-reset').onclick=()=>{state.records=[];state.selectedId=null;state.search='';state.filter='all';editing=false;save();d.close();render();notify('Workspace cleared.');};
  }
  if(action==='classify'){
    busy=true;render();const pending=state.records.filter(r=>!r.decision&&!canReview(r));
    try{const result=await requestPredictions(pending.map(r=>r.product));const rows=alignResponse(result,pending.map(r=>r.product));for(let i=0;i<pending.length;i++){pending[i].result=rows[i];pending[i].mode=health.mode;pending[i].engine_version=result.engine_version;pending[i].relationship_registry_version=result.relationship_registry_version;}save();notify(health.mode==='demo'?'Models are still disconnected. Your products remain saved.':'Predictions updated.');}catch(e){notify(e.message);}finally{busy=false;render();}
  }
});
document.addEventListener('input',e=>{
  if(e.target.id==='search'){state.search=e.target.value;$('#product-table').innerHTML=tableHTML();$('.panel-bottom>span').textContent=`${visibleRecords().length} of ${state.records.length} products`;}
  if(e.target.id==='reviewer'){state.reviewer=e.target.value;save();}
});
document.querySelectorAll('dialog').forEach(d=>d.addEventListener('cancel',e=>{if(busy)e.preventDefault();}));

async function init(){
  try{
    const initialized=await transport.initialize();health=initialized.health;examples=initialized.examples;
    let restored=false;
    try{const stored=JSON.parse(localStorage.getItem(KEY));if(stored?.version===1&&Array.isArray(stored.records)&&stored.records.every(r=>r.product?.record_id&&r.source&&r.result)){state={...state,...stored};restored=true;}}catch{storageOK=false;}
    if(!restored&&health.mode==='demo'){
      state.records=newRecords(examples.request.products.map(product=>({product,source:{...product}})),examples.response,'demo','Supplied examples');state.selectedId=state.records[0].product.record_id;save();
    }
    render();
  }catch(e){$('#app').innerHTML=`<div class="empty"><h1>Couldn’t open the workspace</h1><p>${esc(e.message)} Start the local server, then reload this page.</p><button class="button" id="reload">Try again</button></div>`;$('#reload').onclick=()=>location.reload();}
}
init();
