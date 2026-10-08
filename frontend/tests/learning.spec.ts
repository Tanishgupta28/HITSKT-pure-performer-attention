import { test, expect } from "@playwright/test";

test("register, diagnose, resume, review progress, then practice on mobile", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Start something good." }),
  ).toBeVisible();
  await page.getByLabel("Your name").fill("Preview Learner");
  await page
    .getByLabel("Email address")
    .fill(`browser-${Date.now()}@example.com`);
  await page
    .getByLabel("Password", { exact: true })
    .fill("learning-browser-test-42");
  await page.getByRole("button", { name: "Create my learning space" }).click();
  await expect(
    page.getByRole("heading", { name: "Hello, Preview" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Your concept map" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/dashboard-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Find my starting point" }).click();
  await expect(
    page.getByRole("group", { name: "Answer choices" }),
  ).toBeVisible();
  await page.locator(".choice").first().click();
  await page.getByRole("button", { name: "Check my answer" }).click();
  await expect(page.getByRole("status")).toBeVisible();
  await page.getByRole("button", { name: "Save & leave" }).click();
  await expect(
    page.getByRole("button", { name: "Continue assessment" }),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Continue assessment" }).click();
  await expect(page.getByText("2 of 12", { exact: true })).toBeVisible();
  for (let i = 1; i < 12; i++) {
    await page
      .locator(".choice")
      .nth(i % 4)
      .click();
    await page.getByRole("button", { name: "Check my answer" }).click();
    await expect(page.getByRole("status")).toBeVisible();
    await page
      .getByRole("button", {
        name: i === 11 ? "See my discoveries" : "Next question",
      })
      .click();
  }
  await expect(
    page.getByRole("heading", { name: "That’s progress." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Back to my learning space" }).click();
  await expect(page.getByText("1 ASSESSMENTS COMPLETED")).toBeVisible();
  await page.getByRole("button", { name: "My progress", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Recent learning moments" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", {
      name: "Estimated understanding after each completed assessment",
    }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/progress-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page
    .getByRole("button", { name: "My learning path", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "A path that grows with you" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/path-mobile.png",
    fullPage: true,
  });
  const dimensions = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    width: innerWidth,
  }));
  expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width);
  await page.getByRole("button", { name: "Review & practice" }).first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "Practice this concept" }).click();
  await expect(page.getByText("1 of 6", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/assessment-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
