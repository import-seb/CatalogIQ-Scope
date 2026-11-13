import { test,expect } from '@playwright/test';
import fs from 'node:fs/promises';

test('example review, correction, persistence and both export formats',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Your product workspace'})).toBeVisible();
  await expect(page.getByRole('button',{name:/Claritin RediTabs demo-claritin/})).toBeVisible();
  await page.evaluate(()=>document.fonts.ready);
  await page.screenshot({path:'test-results/workspace-desktop.png',fullPage:true});
  await page.getByRole('button',{name:'Correct',exact:true}).click();
  const select=page.getByLabel('Correct Segment',{exact:true});
  const alternate=await select.locator('option').evaluateAll(options=>options.find(o=>!o.selected).value);
  await select.selectOption(alternate);
  await page.getByLabel('Reviewed by').fill('Maksim');
  await page.getByLabel('Review note',{exact:false}).fill('Checked against the source listing.');
  await page.getByRole('button',{name:'Save review',exact:true}).click();
  await expect(page.locator('.decision-banner')).toContainText('Corrected by Maksim');
  await expect(page.locator('.prediction').first()).toContainText('Changed from Allergy');
  await page.reload();await expect(page.locator('.decision-banner')).toContainText('Corrected by Maksim');
  await page.getByRole('button',{name:'Export results'}).click();await page.getByRole('radio',{name:/Full review record/}).check();
  const downloadPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Download export'}).click();
  const artifact=JSON.parse(await fs.readFile(await(await downloadPromise).path(),'utf8'));
  expect(artifact.records[0].result.predictions.Segment.predicted_label).toBe('Allergy');
  expect(artifact.records[0].decision.labels.Segment).toBe(alternate);expect(artifact.records[0].history).toHaveLength(1);expect(artifact.records[0].mode).toBe('demo');
  await page.getByRole('button',{name:'Export results'}).click();await page.getByLabel('Only approved and corrected products').check();
  const csvPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Download export'}).click();
  const csv=await fs.readFile(await(await csvPromise).path(),'utf8');expect(csv).toContain('demo-claritin-001');expect(csv).not.toContain('demo-culturelle-002');
  expect(errors).toEqual([]);
});

test('CSV import keeps IDs and extra columns; new products never get fake predictions',async({page})=>{
  await page.goto('/');await page.getByRole('button',{name:'Add products',exact:true}).click();
  const csv='record_id,ProductCategory,ProductName,ProductBrand,ProductDescription,ProductContents,Extra\r\n000007,Health,"<img src=x onerror=alert(1)>",Brand,"Line 1\nLine 2",,"=HYPERLINK(""https://example.org"")"';
  await page.getByLabel('Upload product CSV').setInputFiles({name:'store.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
  await expect(page.locator('#file-ready')).toContainText('1 products ready');
  await page.locator('#import-submit').click();
  await expect(page.locator('#input-dialog')).not.toBeVisible();
  await expect(page.locator('.review-product')).toContainText('000007');
  await expect(page.locator('.review-product')).toContainText('<img src=x onerror=alert(1)>');
  await expect(page.getByRole('button',{name:'Approve',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Try classification'})).toBeEnabled();
  const source=await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.at(-1));
  expect(source.source.Extra).toBe('=HYPERLINK("https://example.org")');expect(source.product.Extra).toBeUndefined();expect(source.product.record_id).toBe('000007');
  expect(Object.values(source.result.predictions).every(p=>p.confidence===null)).toBe(true);
  await page.getByRole('button',{name:'Try classification'}).click();await expect(page.getByRole('button',{name:'Try classification'})).toBeEnabled();
  await page.reload();await expect(page.locator('.review-product')).toContainText('000007');
});

test('invalid files and duplicate IDs leave existing workspace unchanged',async({page})=>{
  await page.goto('/');await page.getByRole('button',{name:'Add products',exact:true}).click();
  await page.getByLabel('Upload product CSV').setInputFiles({name:'bad.csv',mimeType:'text/csv',buffer:Buffer.from('record_id,ProductName\n1,Wrong')});
  await expect(page.locator('#input-error')).toContainText('Missing column');await expect(page.locator('#import-submit')).toBeDisabled();
  await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('demo-claritin-001');await page.getByLabel('Product name',{exact:true}).fill('Different product');await page.locator('#import-submit').click();
  await expect(page.locator('#input-error')).toContainText('already exists');
  expect(await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.length)).toBe(2);
});

test('reject requires a reason and reviewer; reopening preserves the audit trail',async({page})=>{
  await page.goto('/');await page.getByRole('button',{name:'Reject',exact:true}).click();await expect(page.locator('#review-error')).toContainText('Enter your name');
  await page.getByLabel('Reviewed by').fill('Max');await page.getByRole('button',{name:'Reject',exact:true}).click();await expect(page.locator('#review-error')).toContainText('short reason');
  await page.getByLabel('Review note',{exact:false}).fill('Listing needs clarification.');await page.getByRole('button',{name:'Reject',exact:true}).click();await expect(page.locator('.decision-banner')).toContainText('Rejected by Max');
  await page.getByRole('button',{name:'Review again'}).click();await page.getByRole('button',{name:'Save review',exact:true}).click();await expect(page.locator('.decision-banner')).toContainText('Approved by Max');
  await page.reload();expect(await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records[0].history.length)).toBe(2);
});

test('mobile layout, search, clear confirmation and single product entry',async({page})=>{
  await page.setViewportSize({width:390,height:844});await page.goto('/');await expect(page.locator('.review-product')).toBeVisible();
  await page.screenshot({path:'test-results/workspace-mobile.png',fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByLabel('Search products').fill('Culturelle');await expect(page.locator('tbody tr')).toHaveCount(1);
  await page.getByRole('button',{name:'Clear workspace',exact:true}).click();await page.getByRole('button',{name:'Keep products'}).click();await expect(page.locator('tbody tr')).toHaveCount(1);
  await page.getByRole('button',{name:'Clear workspace',exact:true}).click();await page.locator('#confirm-reset').click();await expect(page.getByRole('heading',{name:'Your next catalog starts here'})).toBeVisible();
  await page.getByRole('button',{name:'Add products',exact:true}).first().click();await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('00042');await page.getByLabel('Product name',{exact:true}).fill('New sample product');await page.locator('#import-submit').click();await expect(page.locator('.review-product')).toContainText('00042');
  await page.reload();await expect(page.locator('.review-product')).toContainText('00042');
});

test('a mismatched adapter response cannot attach results to another product',async({page})=>{
  await page.goto('/');await page.route('**/api/predict',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({schema_version:'catalogiq-inference-request-v1',products:[]})}));
  await page.getByRole('button',{name:'Add products',exact:true}).click();await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('new-product');await page.getByLabel('Product name',{exact:true}).fill('New product');await page.locator('#import-submit').click();await expect(page.locator('#input-error')).toContainText('does not match');
  expect(await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.length)).toBe(2);
});

test('partial model failures stay explicit and cannot be approved',async({page})=>{
  await page.goto('/');
  const examples=await(await page.request.get('/api/examples')).json();
  await page.route('**/api/predict',async route=>{
    const request=route.request().postDataJSON();const response=structuredClone(examples.response);response.products=[response.products[0]];response.products[0].record_id=request.products[0].record_id;
    Object.assign(response.products[0].predictions.Brand,{status:'failed',predicted_label:null,confidence:null,probabilities:null,review_reasons:['prediction_failed'],error:'Internal file path must never appear in UI'});
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(response)});
  });
  await page.getByRole('button',{name:'Add products',exact:true}).click();await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('partial-001');await page.getByLabel('Product name',{exact:true}).fill('Partial result');await page.locator('#import-submit').click();
  await expect(page.locator('.review-callout')).toContainText('Some predictions are unavailable');await expect(page.locator('#review-panel')).toContainText('Couldn’t classify');await expect(page.getByRole('button',{name:'Approve',exact:true})).toBeDisabled();await expect(page.locator('body')).not.toContainText('Internal file path');
});

test('network failure keeps entry intact and offers a clear retry',async({page})=>{
  await page.goto('/');await page.route('**/api/predict',route=>route.abort());
  await page.getByRole('button',{name:'Add products',exact:true}).click();await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('retry-001');await page.getByLabel('Product name',{exact:true}).fill('Keep my input');await page.locator('#import-submit').click();
  await expect(page.locator('#input-error')).toContainText('Couldn’t reach the local server');await expect(page.getByLabel('Product name',{exact:true})).toHaveValue('Keep my input');await expect(page.locator('#import-submit')).toBeEnabled();
  await page.unroute('**/api/predict');await page.locator('#import-submit').click();await expect(page.locator('.review-product')).toContainText('Keep my input');
});
