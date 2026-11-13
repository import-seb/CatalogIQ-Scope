import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createTransport,demoPredictions} from '../public/transport.js';
const request=JSON.parse(fs.readFileSync(new URL('../../docs/integration/inference_adapter_request.json',import.meta.url)));
const response=JSON.parse(fs.readFileSync(new URL('../../docs/integration/inference_adapter_response.json',import.meta.url)));
const examples={request,response};

test('static mode loads only sample JSON; uploads never use a network request',async()=>{
  const calls=[];
  const transport=createTransport({staticDemo:true,fetcher:async(...args)=>{calls.push(args);return {ok:true,json:async()=>examples};}});
  const initialized=await transport.initialize();assert.equal(initialized.health.browser_only,true);
  const products=structuredClone(request.products);products[0].ProductName='A new private product';
  const result=await transport.predict(products);
  assert.deepEqual(calls,[['/examples.json']]);assert.equal(result.products[0].predictions.Segment.confidence,null);assert.equal(result.products[1].predictions.Segment.predicted_label,'Digestive Health');
});
test('exact sample outputs are preserved without mutating fixtures',()=>{
  const products=structuredClone(request.products);products[0].record_id='0001';
  const result=demoPredictions(products,examples);assert.equal(result.products[0].record_id,'0001');assert.deepEqual(result.products[0].predictions,response.products[0].predictions);assert.equal(examples.response.products[0].record_id,'demo-claritin-001');
});
test('static demo rejects malformed batches and never predicts quarantined products',()=>{
  const products=structuredClone(request.products);products[0].quarantined=true;
  assert.equal(demoPredictions(products,examples).products[0].predictions.Segment.predicted_label,null);
  assert.throws(()=>demoPredictions([],examples));
  const duplicate=structuredClone(request.products);duplicate[1].record_id=duplicate[0].record_id;assert.throws(()=>demoPredictions(duplicate,examples));
  products[0].Segment='Injected truth';assert.throws(()=>demoPredictions(products,examples));
});
test('live/local mode retains the exact adapter envelope',async()=>{
  let sent;
  const transport=createTransport({fetcher:async(url,options)=>{sent={url,body:JSON.parse(options.body)};return {ok:true,json:async()=>response};}});
  assert.deepEqual(await transport.predict(request.products),response);assert.equal(sent.url,'/api/predict');assert.deepEqual(sent.body,request);
});
