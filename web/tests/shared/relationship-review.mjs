import fs from 'node:fs/promises';
import {savedFiles, toCSV, warningHeaders} from './saved-result-fixture.mjs';

async function openImport(page, files) {
  await page.getByRole('button', {name: 'Add products', exact: true}).click();
  await page.getByRole('button', {name: 'Saved results', exact: true}).click();
  await page.getByLabel('Predictions CSV', {exact: true}).setInputFiles({name: 'target_predictions.csv', mimeType: 'text/csv', buffer: Buffer.from(files.predictionsCSV)});
  await page.getByLabel('Relationship warnings CSV', {exact: true}).setInputFiles({name: 'relationship_conflicts.csv', mimeType: 'text/csv', buffer: Buffer.from(files.warningsCSV)});
  await page.getByLabel('Classification report JSON', {exact: true}).setInputFiles({name: 'target_classification_report.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(files.report))});
  await page.getByRole('button', {name: 'Read saved results', exact: true}).click();
}

export function relationshipReviewTests(test, expect) {
  test('saved warnings stay informational through approval, correction, refresh and export', async ({page}, testInfo) => {
    const posted = [];
    const errors = [];
    page.on('request', request => { if (request.method() !== 'GET') posted.push(request.url()); });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    const files = savedFiles({missing: true});
    files.warnings[0].child_prediction = 'Acid Relief';
    files.warningsCSV = toCSV([...files.warnings].reverse(), warningHeaders);
    await openImport(page, files);
    await expect(page.locator('#saved-results-summary')).toContainText('2 relationship warnings across 2 products');
    await page.getByRole('button', {name: 'Add selected batch', exact: true}).click();
    await expect(page.getByRole('region', {name: 'Relationship warnings'})).toContainText('may not belong together');
    await expect(page.getByRole('button', {name: 'Approve', exact: true})).toBeEnabled();
    await page.getByLabel('Reviewed by').fill('Test reviewer');
    await page.getByRole('button', {name: 'Approve', exact: true}).click();
    await expect(page.locator('.decision-banner')).toContainText('Approved by Test reviewer');
    await expect(page.getByRole('region', {name: 'Relationship warnings'})).toContainText('original model suggestions');
    await page.getByRole('button', {name: 'Review again', exact: true}).click();
    await page.getByLabel('Correct Sub-segment', {exact: true}).selectOption('Oral Antihistamine');
    await page.getByLabel('Review note', {exact: false}).fill('Checked the listing.');
    await page.getByRole('button', {name: 'Save review', exact: true}).click();
    await page.reload();
    await expect(page.locator('.decision-banner')).toContainText('Corrected by Test reviewer');
    await expect(page.getByRole('region', {name: 'Relationship warnings'})).toBeVisible();
    await page.getByRole('button', {name: 'Export results', exact: true}).click();
    await page.getByRole('radio', {name: /Full review record/}).check();
    const downloaded = page.waitForEvent('download');
    await page.getByRole('button', {name: 'Download export', exact: true}).click();
    const exported = JSON.parse(await fs.readFile(await (await downloaded).path(), 'utf8'));
    const record = exported.records.find(record => record.product.record_id === '000001');
    expect(record.result.predictions['Sub-Segment'].predicted_label).toBe('Acid Relief');
    expect(record.decision.labels['Sub-Segment']).toBe('Oral Antihistamine');
    expect(record.history).toHaveLength(2);
    expect(record.relationship_warnings[0].is_approved_decision).toBe(false);
    expect(record.relationship_import.files).toHaveLength(3);
    expect(record.relationship_import.files[0].sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(record.mode).toBe('imported');
    expect(record.result.predictions.Brand.probabilities).toBeNull();
    await page.getByRole('button', {name: 'Export results', exact: true}).click();
    const csvDownloaded = page.waitForEvent('download');
    await page.getByRole('button', {name: 'Download export', exact: true}).click();
    const csv = await fs.readFile(await (await csvDownloaded).path(), 'utf8');
    expect(csv).toContain('invalid_approved_classification_relationship');
    expect(csv).toContain('catalogiq:relationship_import');
    await page.setViewportSize({width: 390, height: 844});
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({path: testInfo.outputPath('relationship-mobile.png'), fullPage: true});
    await page.getByRole('button', {name: 'Unnamed product 000002', exact: true}).click();
    await expect(page.getByRole('region', {name: 'Relationship warnings'})).toContainText('classification needed to check this relationship is missing');
    await expect(page.getByRole('button', {name: 'Approve', exact: true})).toBeDisabled();
    expect(posted).toEqual([]);
    expect(errors).toEqual([]);
  });

  test('stale warnings and a failed browser save preserve the existing workspace', async ({page}) => {
    await page.goto('/');
    const files = savedFiles();
    await openImport(page, files);
    await expect(page.getByRole('button', {name: 'Add selected batch', exact: true})).toBeEnabled();
    await page.getByLabel('Relationship warnings CSV', {exact: true}).setInputFiles({name: 'old.csv', mimeType: 'text/csv', buffer: Buffer.from(toCSV([], warningHeaders))});
    await expect(page.getByRole('button', {name: 'Add selected batch', exact: true})).toBeDisabled();
    await page.getByRole('button', {name: 'Read saved results', exact: true}).click();
    await expect(page.locator('#saved-results-error')).toContainText('incomplete');
    await page.getByLabel('Relationship warnings CSV', {exact: true}).setInputFiles({name: 'current.csv', mimeType: 'text/csv', buffer: Buffer.from(files.warningsCSV)});
    await page.getByRole('button', {name: 'Read saved results', exact: true}).click();
    await expect(page.getByRole('button', {name: 'Add selected batch', exact: true})).toBeEnabled();
    await page.evaluate(() => {
      const original = Storage.prototype.setItem;
      Storage.prototype.setItem = function (key, value) {
        if (String(value).includes('"mode":"imported"')) throw new DOMException('Full', 'QuotaExceededError');
        return original.call(this, key, value);
      };
    });
    await page.getByRole('button', {name: 'Add selected batch', exact: true}).click();
    await expect(page.locator('#saved-results-error')).toContainText('storage is full');
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.length)).toBe(2);
    await page.getByRole('button', {name: 'Cancel', exact: true}).click();
    await expect(page.locator('tbody tr')).toHaveCount(2);
  });

  test('a large saved run offers selectable batches within workspace capacity', async ({page}) => {
    await page.goto('/');
    await openImport(page, savedFiles({count: 700, conflicts: 651}));
    await expect(page.locator('#saved-results-summary')).toContainText('651 relationship warnings across 651 products');
    await expect(page.getByLabel('Batch', {exact: true}).locator('option')).toHaveCount(3);
    await page.getByLabel('Batch', {exact: true}).selectOption('248');
    await page.getByRole('button', {name: 'Add selected batch', exact: true}).click();
    await expect(page.locator('.review-product')).toContainText('000249');
    expect(await page.evaluate(() => JSON.parse(localStorage.getItem('catalogiq-workspace-v1')).records.length)).toBe(250);
    await page.reload();
    await expect(page.locator('.review-product')).toContainText('000249');
  });
}
