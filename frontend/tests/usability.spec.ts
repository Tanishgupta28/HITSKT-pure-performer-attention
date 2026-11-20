import { test, expect, type Page } from "@playwright/test";

async function newLearner(page: Page) {
  await page.goto("/signup");
  await page.getByLabel("Name", { exact: true }).fill("Usability Learner");
  await page
    .getByLabel("Email address")
    .fill(`usability-${crypto.randomUUID()}@example.com`);
  await page.getByLabel("Password", { exact: true }).fill("test-learner-1234");
  await page
    .getByRole("button", { name: "Create account", exact: true })
    .click();
  await page.getByLabel("Grade / class").selectOption("Grade 8");
  await page.getByRole("button", { name: "Continue", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Good to see you, Usability" }),
  ).toBeVisible();
}
async function practice(page: Page) {
  await newLearner(page);
  await page
    .getByRole("button", { name: "Learn", exact: true })
    .first()
    .click();
  await page
    .getByRole("button", { name: "Practise fractions", exact: true })
    .click();
  await expect(
    page.getByRole("group", { name: "Answer choices" }),
  ).toBeVisible();
}

test("four student destinations and old links reach one learning plan and one progress view", async ({
  page,
}) => {
  await newLearner(page);
  await expect(
    page.getByRole("navigation", { name: "Workspace" }).getByRole("button"),
  ).toHaveText(["Today", "Learn", "Assessments", "Progress"]);
  await page.goto("/?view=revision");
  await expect(
    page.getByRole("heading", { name: "Your learning plan" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/refined-learn-desktop.png",
    fullPage: true,
  });
  for (const width of [1024, 768, 390, 360]) {
    await page.setViewportSize({ width, height: 860 });
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
  }
  await page.screenshot({
    path: "../.platform-runtime/refined-learn-mobile.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1080 });
  await page.getByRole("button", { name: "Review lesson" }).first().click();
  await expect(page.getByRole("dialog")).toContainText("Worked example");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Cancel", exact: true })
    .click();
  await page.goto("/?view=history");
  await expect(
    page.getByRole("button", { name: "Session history", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByText("No sessions to show", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Learning report", exact: true })
    .click();
  await expect(
    page.getByText("No learning evidence yet", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Assessments", exact: true }).click();
  await expect(page.getByRole("button", { name: "Start mock" })).toHaveCount(0);
});

test("practice preserves draft and working through refresh and offline, retries an uncertain save once, and finishes", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await practice(page);
  await page.screenshot({
    path: "../.platform-runtime/refined-practice-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("radio").first().check();
  await page.getByText("Working space", { exact: false }).click();
  await page
    .getByLabel("Your calculations")
    .fill("Find a common denominator first.");
  await page.reload();
  await expect(page.getByRole("radio").first()).toBeChecked();
  await page.locator(".p-working-space summary").click();
  await expect(page.getByLabel("Your calculations")).toHaveValue(
    "Find a common denominator first.",
  );
  await page.context().setOffline(true);
  await expect(page.getByText("You’re offline", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Check answer", exact: true }),
  ).toBeDisabled();
  await page.context().setOffline(false);
  await expect(
    page.getByRole("button", { name: "Check answer", exact: true }),
  ).toBeEnabled();
  let dropped = false;
  const requests: string[] = [];
  await page.route("**/api/assessments/*/answers", async (route) => {
    requests.push(route.request().postDataJSON().request_id);
    if (!dropped) {
      dropped = true;
      await route.fetch();
      await route.abort("failed");
    } else await route.continue();
  });
  await page.getByRole("button", { name: "Check answer", exact: true }).click();
  await expect(
    page.getByText("Action could not be completed", { exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("radio").first()).toBeChecked();
  await expect(page.getByRole("radio").first()).toBeDisabled();
  await page.getByRole("button", { name: "Retry answer", exact: true }).click();
  await expect(page.getByText("Answer saved", { exact: true })).toBeVisible();
  expect(requests).toHaveLength(2);
  expect(requests[0]).toBe(requests[1]);
  const id = page.url().split("/").pop();
  const state = await (await page.request.get(`/api/assessments/${id}`)).json();
  expect(state.answered).toBe(1);
  expect(
    await page.evaluate(() => {
      const feedback = document
        .querySelector(".p-answer-feedback")!
        .getBoundingClientRect();
      const footer = document
        .querySelector(".p-practice-footer")!
        .getBoundingClientRect();
      return feedback.bottom <= footer.top;
    }),
  ).toBe(true);
  await page.screenshot({
    path: "../.platform-runtime/refined-practice-mobile.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Next question", exact: true })
    .click();
  await expect(page.getByRole("radio").first()).not.toBeChecked();
  for (let i = 1; i < 6; i++) {
    await page.getByRole("radio").first().check();
    await page
      .getByRole("button", { name: "Check answer", exact: true })
      .click();
    await expect(page.getByText("Answer saved", { exact: true })).toBeVisible();
    await page
      .getByRole("button", {
        name: i === 5 ? "See practice report" : "Next question",
        exact: true,
      })
      .click();
  }
  await expect(
    page.getByRole("heading", { name: "Practice complete", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Review answers", exact: true })
    .click();
  await expect(page.locator(".p-review-list details")).toHaveCount(6);
  expect(errors).toEqual([]);
  await expect
    .poll(() =>
      page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
    )
    .toBe(true);
});

test("save and leave confirms the currently selected diagnostic answer", async ({
  page,
}) => {
  await newLearner(page);
  await page
    .getByRole("button", { name: "Start diagnostic", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Begin diagnostic", exact: true })
    .click();
  await page.getByRole("radio").first().check();
  await page.getByRole("button", { name: "Save & leave", exact: true }).click();
  await page
    .getByRole("button", { name: "Leave assessment", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Resume diagnostic", exact: true })
    .click();
  await expect(page.getByRole("radio").first()).toBeChecked();
  await expect(
    page.getByRole("button", { name: "Question 1, answered", exact: true }),
  ).toBeVisible();
  await page.getByRole("radio").nth(1).check();
  await page
    .getByRole("button", { name: "Question 2, unanswered", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Question 1, answered", exact: true })
    .click();
  await expect(page.getByRole("radio").nth(1)).toBeChecked();
});
