import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: 2,
  retries: 0,
  use: { baseURL:'http://127.0.0.1:8765', channel:process.env.PLAYWRIGHT_CHANNEL || undefined, viewport:{width:1440,height:1050}, trace:'retain-on-failure' },
  webServer: {command:'python server.py --port 8765',url:'http://127.0.0.1:8765/api/health',reuseExistingServer:true},
});
