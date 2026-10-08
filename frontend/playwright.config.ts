import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 20_000 },
  use: {
    actionTimeout: 20_000,
    baseURL: "http://localhost:3000",
    browserName: "chromium",
    channel: process.env.PLAYWRIGHT_CHANNEL || "chromium",
    viewport: { width: 1440, height: 1080 },
    trace: "retain-on-failure",
  },
  reporter: "list",
});
