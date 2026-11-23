import { expect, test, type Page } from "@playwright/test";
import type { JunyiPaper, JunyiRequirement } from "../src/lib/junyi";

const base = process.env.LUMEN_JUNYI_TEST_URL;
test.skip(!base, "Start the isolated Junyi browser fixture before running this test.");

async function signIn(page: Page, email: string) {
  await page.goto(`${base}/login`);
  await page.getByLabel("Email address").fill(email);
  await page.getByLabel("Password", { exact: true }).fill("lumen-fixture-2026");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
}

test("sign-in has a safe fallback before hydration", async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto(`${base}/login`);
    const button = page.getByRole("button", { name: "Sign in", exact: true });
    await expect(button).toBeDisabled();
    await expect(page.locator("form")).toHaveAttribute("method", "post");
  } finally {
    await context.close();
  }
});

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
  await expect(page.getByText("Checking study resources…", {exact: true})).toHaveCount(0);
  const available = page.getByRole("button", { name: "Mark complete" });
  for (let remaining = await available.count(); remaining > 0; remaining--) {
    await available.first().click();
    await expect(available).toHaveCount(remaining - 1);
  }
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

  const answerRoute = "**/api/assessments/*/answers";
  const written = page.waitForRequest((request) => request.url().endsWith("/answers") && request.method() === "POST");
  // Let the server save question one, then simulate a connection drop before the browser hears back.
  await page.route(answerRoute, async (route) => {
    const committed = await route.fetch();
    expect(committed.ok()).toBeTruthy();
    expect((await committed.json()).paper.answered).toBe(1);
    await route.abort("failed");
  }, { times: 1 });
  await page.getByRole("button", { name: "Submit answer", exact: true }).click();
  await expect(page.getByRole("button", { name: "Retry answer", exact: true })).toBeEnabled();
  const replay = page.waitForResponse((response) => response.url().endsWith("/answers") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Retry answer", exact: true }).click();
  const retried = await replay;
  expect(retried.request().postData()).toBe((await written).postData());
  expect((await retried.json()).replayed).toBe(true);

  for (let question = 2; question <= 20; question++) {
    await expect(page.getByText(`Question ${question} of 20`)).toBeVisible();
    await page.getByRole("radio").first().check();
    await page.getByRole("button", { name: "Submit answer" }).click();
  }
  await expect(page.getByRole("heading", { name: "Assessment complete" })).toBeVisible();
  await expect(page.getByText("Actual paper score")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Skill estimates after this paper" })).toBeVisible();
  await expect(page.getByTestId("junyi-evidence-count")).toContainText("20 saved Lumen assessment answers");
  await page.getByLabel("Skill status").selectOption("lumen");
  await expect(page.getByText(/^Lumen \d+\/\d+$/).first()).toBeVisible();
  await page.getByRole("button", { name: "Start next assessment" }).click();
  await expect(page).not.toHaveURL(paperUrl);
});

test("a delayed study response cannot roll back the current checklist", async ({ page }) => {
  const id = "88657e02-d02f-46bf-adef-a157fce332fd";
  const path = `/api/assessments/${id}`;
  const requirements: JunyiRequirement[] = [1, 2, 3].map((tag) => ({
    tag, topic: `Fixture skill ${tag}`, status: "pending", lookup: "verified", can_retry: false, message: null,
    resources: [{url: `https://example.test/lesson/${tag}`, title: `Fixture lesson ${tag}`, source: "fixture"}],
  }));
  const paper: JunyiPaper = {
    id, sequence: 2, revision: 1, status: "locked", stage: "junior", total: 20, answered: 0, message: null,
    requirements, question: null, results: null, analysis_status: null, analysis_retry: false, answers_used: null, skills: [], can_retry: false,
  };
  const snapshot = (revision: number, completed: number): JunyiPaper => ({
    ...paper, revision, requirements: requirements.map((item) => item.tag <= completed ? {...item, status: "satisfied"} : item),
  });
  let releaseOlder = () => {};
  const older = new Promise<void>((resolve) => { releaseOlder = resolve; });
  await page.route(`**${path}`, (route) => route.fulfill({ json: paper }));
  await page.route(`**${path}/study/1/complete`, async (route) => {
    await older;
    await route.fulfill({ json: snapshot(2, 1) });
  });
  await page.route(`**${path}/study/2/complete`, (route) => route.fulfill({ json: snapshot(3, 2) }));
  await page.goto(`${base}/assessment/junyi/${id}`);
  const markComplete = (skill: number) => page.locator(".junyi-study-card").filter({hasText: `Fixture skill ${skill}`})
    .getByRole("button", {name: "Mark complete"});
  await markComplete(1).click();
  await markComplete(2).click();
  await expect(page.getByText("1 skill left to unlock", {exact: true})).toBeVisible();
  const lateResponse = page.waitForResponse((response) => response.url().endsWith("/study/1/complete"));
  releaseOlder();
  await lateResponse;
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  await expect(page.getByText("1 skill left to unlock", {exact: true})).toBeVisible();
  await expect(page.getByText("Study already checked off (2)", {exact: true})).toBeVisible();
  await expect(markComplete(3)).toBeVisible();
});
