import { expect, test, type Page } from "@playwright/test";

const base = process.env.LUMEN_JUNYI_TEST_URL;
test.skip(!base, "Start the isolated Junyi browser fixture before running this test.");

async function signIn(page: Page, email: string) {
  await page.goto(`${base}/login`);
  await page.getByLabel("Email address").fill(email);
  await page.getByLabel("Password", { exact: true }).fill("lumen-fixture-2026");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
}

test("assigned learner can unlock, resume, submit and review a Junyi paper", async ({ page, browser }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await signIn(page, "junyi-browser@example.com");
  await expect(page.getByRole("heading", { name: "Junyi Adaptive Assessment" })).toBeVisible();
  await page.getByRole("button", { name: "Start adaptive assessment" }).click();
  await expect(page.getByRole("heading", { name: "Study before your assessment" })).toBeVisible();
  const paperUrl = page.url();
  await expect(page.locator(".junyi-study-card").first()).toBeVisible();
  await expect(page.getByRole("radiogroup")).toHaveCount(0);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Study before your assessment" })).toBeVisible();
  expect(page.url()).toBe(paperUrl);
  await expect(page.getByRole("button", { name: "Mark complete" }).first()).toBeVisible();
  const available = page.getByRole("button", { name: "Mark complete" });
  while (await available.count()) await available.first().click();
  const missing = page.locator(".junyi-study-card").filter({ hasText: "This fixture has no verified resource." }).first();
  await expect(missing).toBeVisible();
  await missing.getByRole("button", { name: "Retry resource search" }).click();
  await expect(missing.getByRole("button", { name: "Retry resource search" })).toBeVisible();
  await expect(page.getByRole("radiogroup")).toHaveCount(0);

  const admin = await browser.newPage();
  try {
    await signIn(admin, "junyi-operator@example.com");
    await expect(admin.getByRole("heading", { name: "Junyi assessment operations" })).toBeVisible();
    const overrides = admin.getByRole("button", { name: "Clear requirement" });
    await expect(overrides.first()).toBeVisible();
    while (await overrides.count()) {
      const previous = await overrides.count();
      await overrides.first().click();
      await admin.getByLabel("Reason for clearing the requirement").fill("Fixture resource unavailable");
      await admin.getByRole("button", { name: "Confirm override" }).click();
      await expect(overrides).toHaveCount(previous - 1);
    }
  } finally {
    await admin.close();
  }

  await page.reload();
  await expect(page.getByText("Question 1 of 20")).toBeVisible();
  const prompt = page.getByRole("heading", { level: 1 });
  const first = await prompt.textContent();
  await page.getByRole("radio").first().check();
  await page.reload();
  await expect(prompt).toHaveText(first!);
  await expect(page.getByRole("radio").first()).toBeChecked();
  if (process.env.LUMEN_JUNYI_SCREENSHOT) await page.screenshot({ path: process.env.LUMEN_JUNYI_SCREENSHOT });
  for (let question = 1; question <= 20; question++) {
    await expect(page.getByText(`Question ${question} of 20`)).toBeVisible();
    await page.getByRole("radio").first().check();
    await page.getByRole("button", { name: "Submit answer" }).click();
  }
  await expect(page.getByRole("heading", { name: "Assessment complete" })).toBeVisible();
  await expect(page.getByText("Actual paper score")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Skill estimates after this paper" })).toBeVisible();
  await page.getByRole("button", { name: "Start next assessment" }).click();
  await expect(page).not.toHaveURL(paperUrl);
});
