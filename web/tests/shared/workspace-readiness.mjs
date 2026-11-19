import fs from 'node:fs/promises';

// Exercise the same user workflows against the local API and browser-only demo.
export function readinessTests(test, expect) {
  test('CSV preview preserves values, catches existing IDs and clears stale previews', async ({page}, testInfo) => {
    await page.goto('/');
    await page.getByRole('button',{name:'Add products',exact:true}).click();
    const header='record_id,ProductCategory,ProductName,ProductBrand,ProductDescription,ProductContents,Note';
    const file=csv=>({name:'preview.csv',mimeType:'text/csv',buffer:Buffer.from(csv)});
    await page.getByLabel('Upload product CSV').setInputFiles(file(header+'\ndemo-claritin-001,Health,Existing,Brand,,,Keep'));
    await expect(page.locator('#input-error')).toContainText('already exists');
    await expect(page.locator('#import-submit')).toBeDisabled();
    await expect(page.locator('#csv-preview')).toBeHidden();
    const rows=Array.from({length:7},(_,i)=>`000${i},Health,${i===0?'<img src=x onerror=alert(1)>':'Product '+i},Brand,,,Keep`);
    await page.getByLabel('Upload product CSV').setInputFiles(file(header+'\n'+rows.join('\n')));
    await expect(page.locator('#csv-preview')).toContainText('Showing 5 of 7 products');
    await expect(page.locator('#csv-preview tbody tr')).toHaveCount(5);
    await expect(page.locator('#csv-preview')).toContainText('0000');
    await expect(page.locator('#csv-preview')).toContainText('<img src=x onerror=alert(1)>');
    await expect(page.locator('#csv-preview img')).toHaveCount(0);
    await expect(page.locator('#csv-preview')).toContainText('Note');
    expect(await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.length)).toBe(2);
    await page.screenshot({path:testInfo.outputPath('csv-preview-desktop.png')});
    await page.setViewportSize({width:390,height:844});
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    await page.screenshot({path:testInfo.outputPath('csv-preview-mobile.png')});
    await page.getByLabel('Upload product CSV').setInputFiles(file('record_id,ProductName\n1,Wrong'));
    await expect(page.locator('#input-error')).toContainText('Missing column');
    await expect(page.locator('#csv-preview')).toBeHidden();
    await expect(page.locator('#import-submit')).toBeDisabled();
  });

  test('correction drafts survive product switches and refresh without becoming decisions', async ({page}) => {
    await page.goto('/');
    await page.getByRole('button',{name:'Correct',exact:true}).click();
    const select=page.getByLabel('Correct Segment',{exact:true});
    const label=await select.locator('option').evaluateAll(options=>options.find(o=>!o.selected).value);
    await select.selectOption(label);
    await page.getByLabel('Reviewed by').fill('Draft reviewer');
    await page.getByLabel('Review note',{exact:false}).fill('Still checking the listing.');
    await expect(page.locator('#review-draft-status')).toContainText('Draft saved');
    await page.getByRole('button',{name:/Culturelle Daily Probiotic demo-culturelle/}).click();
    await expect(page.getByLabel('Review note',{exact:false})).toHaveValue('');
    await page.getByRole('button',{name:/Claritin RediTabs demo-claritin/}).click();
    await expect(page.getByLabel('Correct Segment',{exact:true})).toHaveValue(label);
    await page.reload();
    await expect(page.getByLabel('Correct Segment',{exact:true})).toHaveValue(label);
    await expect(page.getByLabel('Review note',{exact:false})).toHaveValue('Still checking the listing.');
    await expect(page.getByLabel('Reviewed by')).toHaveValue('Draft reviewer');
    await page.getByRole('button',{name:'Export results'}).click();
    await page.getByRole('radio',{name:/Full review record/}).check();
    const pending=page.waitForEvent('download');
    await page.getByRole('button',{name:'Download export'}).click();
    const record=JSON.parse(await fs.readFile(await(await pending).path(),'utf8')).records[0];
    expect(record.decision).toBeNull();expect(record.history).toEqual([]);
    expect(record.result.predictions.Segment.predicted_label).toBe('Allergy');
    expect(record.draft.labels.Segment).toBe(label);
    await page.getByRole('button',{name:'Save review',exact:true}).click();
    await expect(page.locator('.decision-banner')).toContainText('Corrected by Draft reviewer');
    const saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records[0]);
    expect(saved.draft).toBeUndefined();expect(saved.history).toHaveLength(1);
    expect(saved.decision.labels.Segment).toBe(label);
  });

  test('discarding a new draft preserves the prior decision and history', async ({page}) => {
    await page.goto('/');
    await page.getByLabel('Reviewed by').fill('Original reviewer');
    await page.getByRole('button',{name:'Approve',exact:true}).click();
    await page.getByRole('button',{name:'Review again'}).click();
    const select=page.getByLabel('Correct Segment',{exact:true});
    const label=await select.locator('option').evaluateAll(options=>options.find(o=>!o.selected).value);
    await select.selectOption(label);
    await page.getByLabel('Review note',{exact:false}).fill('Unfinished correction');
    await page.getByRole('button',{name:'Discard draft',exact:true}).click();
    await page.reload();
    await expect(page.locator('.decision-banner')).toContainText('Approved by Original reviewer');
    const record=await page.evaluate(()=>JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records[0]);
    expect(record.draft).toBeUndefined();expect(record.history).toHaveLength(1);
    expect(record.decision.labels.Segment).toBe('Allergy');
  });

  test('a pending review note survives filter changes and reload', async ({page}) => {
    await page.goto('/');
    await page.getByLabel('Reviewed by').fill('Note reviewer');
    await page.getByLabel('Review note',{exact:false}).fill('Need the original product label.');
    await page.getByRole('button',{name:'Reviewed',exact:true}).click();
    await page.getByRole('button',{name:'Products',exact:true}).click();
    await page.reload();
    await expect(page.getByLabel('Review note',{exact:false})).toHaveValue('Need the original product label.');
    await page.getByRole('button',{name:'Reject',exact:true}).click();
    await expect(page.locator('.decision-banner')).toContainText('Rejected by Note reviewer');
  });
}
