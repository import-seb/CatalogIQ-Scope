import {defineConfig} from '@playwright/test';
const baseURL=process.env.CATALOGIQ_DEMO_URL||'http://127.0.0.1:8766';
export default defineConfig({
  testDir:'./tests/static-browser',fullyParallel:true,workers:2,
  outputDir:'test-results-static',
  use:{baseURL,channel:process.env.PLAYWRIGHT_CHANNEL||undefined,viewport:{width:1440,height:1000},trace:'retain-on-failure'},
  webServer:process.env.CATALOGIQ_DEMO_URL?undefined:{command:'python -m http.server 8766 --bind 127.0.0.1 --directory dist',url:baseURL,reuseExistingServer:true},
});
