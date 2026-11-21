import { test, expect } from "@playwright/test";

test("register and start mixed practice on mobile", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Know what to/ }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/lumen-landing.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Create student account" }).click();
  await page.getByLabel("Name", { exact: true }).fill("QA Learner");
  await page
    .getByLabel("Email address")
    .fill(`browser-${Date.now()}@example.com`);
  await page.getByLabel("Password", { exact: true }).fill("test-learner-1234");
  await page
    .getByRole("button", { name: "Create account", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Set up your learning" }),
  ).toBeVisible();
  await page.getByLabel("Grade level").selectOption("Grade 8");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Good to see you, QA" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Learn", exact: true }).click();
  await page.getByText("Mix topics & plan your week", { exact: true }).click();
  await page
    .getByRole("button", { name: "Start mixed practice", exact: true })
    .click();
  await expect(
    page.getByRole("group", { name: "Answer choices" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Show a hint" }).click();
  await expect(page.getByText("Hint shown · 1 used")).toBeVisible();
  await page.getByRole("radio").first().check();
  await page.getByRole("button", { name: "Check answer" }).click();
  await expect(page.getByText("Answer saved", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/lumen-mobile-practice.png",
    fullPage: true,
  });
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
  expect(errors).toEqual([]);
});
