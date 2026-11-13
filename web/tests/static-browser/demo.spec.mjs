import {test,expect} from '@playwright/test';
import fs from 'node:fs/promises';

test('shared demo supports correction, export and refresh with no API',async({page})=>{
  const errors=[],apiRequests=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
  page.on('request',r=>{if(r.method()!=='GET'||r.url().includes('/api/'))apiRequests.push(r.url());});
  await page.goto('/');await expect(page.getByRole('heading',{name:'Your product workspace'})).toBeVisible();
  await expect(page.locator('.footer-note')).toContainText('Uploads and reviews stay in your browser');
  await page.getByRole('button',{name:'Correct',exact:true}).click();
  const select=page.getByLabel('Correct Segment',{exact:true});const other=await select.locator('option').evaluateAll(o=>o.find(v=>!v.selected).value);await select.selectOption(other);
  await page.getByLabel('Reviewed by').fill('Team demo reviewer');await page.getByLabel('Review note',{exact:false}).fill('Testing a correction in the shared demo.');await page.getByRole('button',{name:'Save review',exact:true}).click();
  await expect(page.locator('.decision-banner')).toContainText('Corrected by Team demo reviewer');
  await page.reload();await expect(page.locator('.decision-banner')).toContainText('Corrected by Team demo reviewer');
  await page.getByRole('button',{name:'Export results'}).click();await page.getByRole('radio',{name:/Full review record/}).check();
  const pending=page.waitForEvent('download');await page.getByRole('button',{name:'Download export'}).click();const file=await pending;
  const json=JSON.parse(await fs.readFile(await file.path(),'utf8'));expect(json.records[0].decision.labels.Segment).toBe(other);expect(json.records[0].result.predictions.Segment.predicted_label).toBe('Allergy');expect(json.records[0].mode).toBe('demo');
  expect(apiRequests).toEqual([]);expect(errors).toEqual([]);
});

test('CSV stays local and creates no invented scores; separate visitors have separate reviews',async({page,browser,baseURL})=>{
  const sent=[],uploadedData=[];page.on('request',r=>{if(r.method()!=='GET')sent.push(r.url());if((r.postData()||'').includes('not-to-upload'))uploadedData.push(r.url());});
  await page.goto('/');await page.getByRole('button',{name:'Add products',exact:true}).click();
  const csv='record_id,ProductCategory,ProductName,ProductBrand,ProductDescription,ProductContents,private_note\n00009,Health,Private new product,Brand,Description,,not-to-upload';
  await page.getByLabel('Upload product CSV').setInputFiles({name:'products.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});await expect(page.locator('#file-ready')).toBeVisible();await page.locator('#import-submit').click();
  await expect(page.locator('.review-product')).toContainText('00009');await expect(page.getByRole('button',{name:'Approve',exact:true})).toBeDisabled();
  await page.getByRole('button',{name:'Try classification'}).click();await expect(page.getByRole('button',{name:'Try classification'})).toBeEnabled();
  const record=await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.at(-1));expect(record.source.private_note).toBe('not-to-upload');expect(Object.values(record.result.predictions).every(p=>p.confidence===null)).toBe(true);expect(sent).toEqual([]);expect(uploadedData).toEqual([]);
  const visitor=await browser.newContext();const second=await visitor.newPage();await second.goto(baseURL);await expect(second.locator('tbody tr')).toHaveCount(2);await expect(second.locator('body')).not.toContainText('Private new product');await visitor.close();
});

test('shared demo works on mobile and allows a single product',async({page})=>{
  await page.setViewportSize({width:390,height:844});await page.goto('/');await expect(page.getByRole('heading',{name:'Your product workspace'})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByRole('button',{name:'Add products',exact:true}).click();await page.getByRole('button',{name:'One product',exact:true}).click();await page.getByLabel('Product ID',{exact:true}).fill('mobile-001');await page.getByLabel('Product name',{exact:true}).fill('Mobile product');await page.locator('#import-submit').click();await expect(page.locator('.review-product')).toContainText('mobile-001');
});
