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
  await page.getByRole("button", { name: "My profile", exact: true }).click();
  await page.getByLabel("Grade or learning stage").selectOption("Grade 9");
  await page.getByLabel("Curriculum").selectOption("ICSE");
  await page
    .getByLabel("My learning goal")
    .fill("Get confident with geometry and everyday maths.");
  await page.getByLabel("7-day question target").fill("18");
  await page.getByLabel("Adaptive session length").selectOption("6");
  await page.getByLabel("Tue", { exact: true }).check();
  await page.getByLabel("Area & perimeter", { exact: true }).check();
  await page.getByRole("button", { name: "Save my preferences" }).click();
  await expect(page.getByRole("status")).toHaveText("Preferences saved");
  await page.reload();
  await page.getByRole("button", { name: "My profile", exact: true }).click();
  await expect(page.getByLabel("Grade or learning stage")).toHaveValue(
    "Grade 9",
  );
  await expect(page.getByLabel("7-day question target")).toHaveValue("18");
  await expect(
    page.getByLabel("Area & perimeter", { exact: true }),
  ).toBeChecked();
  await page.screenshot({
    path: "../.platform-runtime/profile-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "../.platform-runtime/profile-mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await expect(page.getByText("384 QUESTIONS · 4 CONCEPTS")).toBeVisible();
  await expect(
    page.getByRole("progressbar", { name: "Question target progress" }),
  ).toHaveAttribute("aria-valuemax", "18");
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
  const dashboard = await (
    await page.request.get("/api/learning/mathematics")
  ).json();
  expect(dashboard.weekly_goal).toEqual({
    answered: 12,
    target: 18,
    remaining: 6,
  });
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
  await expect(page.getByText("Work through an example")).toBeVisible();
  await page.getByRole("button", { name: "Practice this concept" }).click();
  await expect(page.getByText("1 of 6", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/assessment-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
