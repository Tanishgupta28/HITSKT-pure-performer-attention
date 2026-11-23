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

test("assigned learner can skip missing links, unlock, resume and review a paper", async ({ page }) => {
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
  await expect(page.getByText("Question 1 of 20")).toBeVisible();
  await page.getByText(/Optional study without verified links/).click();
  const missing = page.locator(".junyi-study-card").filter({ hasText: "This fixture has no verified resource." }).first();
  await expect(missing).toBeVisible();
  await expect(missing.getByText("Optional for now", { exact: true })).toBeVisible();
  await missing.getByRole("button", { name: "Retry resource search" }).click();
  await expect(missing.getByRole("button", { name: "Retry resource search" })).toBeVisible();
  await expect(page.getByRole("radiogroup")).toBeVisible();

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
    tag, topic: `Fixture skill ${tag}`, status: "pending", lookup: "verified", can_retry: false, required: true, message: null,
    resources: [{url: `https://example.test/lesson/${tag}`, title: `Fixture lesson ${tag}`, source: "fixture"}],
  }));
  const paper: JunyiPaper = {
    id, sequence: 2, revision: 1, status: "locked", stage: "junior", total: 20, answered: 0, message: null,
    requirements, question: null, results: null, analysis_status: null, analysis_retry: false, answers_used: null, skills: [], can_retry: false,
  };
  const snapshot = (revision: number, completed: number): JunyiPaper => ({
    ...paper, revision, requirements: requirements.map((item) => item.tag <= completed ? {...item, status: "satisfied", required: false} : item),
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

test("a newly verified lesson pauses a paper and preserves the answer draft", async ({ page }) => {
  const id = "bcaedd7c-000c-4c24-8c51-b40d159c8ee1";
  const path = `/api/assessments/${id}`;
  const waiting: JunyiRequirement = {
    tag: 1, topic: "Equivalent fractions", status: "pending", lookup: "unavailable", required: false,
    resources: [], message: "No verified resource", can_retry: true,
  };
  const verified: JunyiRequirement = {
    ...waiting, lookup: "verified", required: true, message: null, can_retry: false,
    resources: [{url: "https://example.test/lesson", title: "Equivalent fractions lesson", source: "fixture"}],
  };
  const paper: JunyiPaper = {
    id, sequence: 2, revision: 1, status: "open", stage: "junior", total: 20, answered: 1, message: null,
    requirements: [waiting], question: {id: "q2", topic: "Fractions", prompt: "Which fraction equals 1/2?",
      options: [{id: "a", text: "2/4"}, {id: "b", text: "3/4"}]},
    results: null, analysis_status: null, analysis_retry: false, answers_used: null, skills: [], can_retry: false,
  };
  let current = paper;
  await page.route(`**${path}`, (route) => route.fulfill({ json: current }));
  await page.route(`**${path}/study/1/complete`, (route) => {
    current = {...paper, revision: 3, requirements: [{...verified, status: "satisfied", required: false}]};
    return route.fulfill({ json: current });
  });
  await page.goto(`${base}/assessment/junyi/${id}`);
  await expect(page.getByText("Question 2 of 20")).toBeVisible();
  await page.getByRole("radio").first().check();
  current = {...paper, revision: 2, status: "locked", question: null, requirements: [verified]};
  await expect(page.getByText("1 skill left to unlock", {exact: true})).toBeVisible();
  await expect(page.getByRole("radiogroup")).toHaveCount(0);
  await page.getByRole("button", {name: "Mark complete"}).click();
  await expect(page.getByText("Question 2 of 20")).toBeVisible();
  await expect(page.getByRole("radio").first()).toBeChecked();
});
