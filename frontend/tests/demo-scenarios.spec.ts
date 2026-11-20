import { expect, test, type Page } from "@playwright/test";

async function signIn(page: Page, handle: string) {
  await page.goto("/login");
  await page
    .getByLabel("Email address")
    .fill(`${handle}@demo.lumen.example.com`);
  await page.getByLabel("Password", { exact: true }).fill("Lumen12345");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText("Synthetic demo", { exact: true })).toBeVisible();
}

test("the demo shows distinct learners, useful practice reviews, and a partly submitted class paper", async ({
  page,
  browser,
}) => {
  const errors: string[] = [];
  await signIn(page, "teacher");
  await page.getByRole("button", { name: "Analytics", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Compare topic evidence" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Asha Shah · 6/12" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Rohan Mehta · 9/12" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/demo-cohort-comparison.png",
    fullPage: true,
  });

  const classList = await (
    await page.request.get("/api/teacher/classes")
  ).json();
  const classId = classList.find(
    (row: { name: string }) => row.name === "8B Mathematics",
  )._id;
  const roster = await (
    await page.request.get(`/api/teacher/classes/${classId}`)
  ).json();
  expect(roster.students).toHaveLength(24);
  expect(roster.diagnostic_count).toBe(18);
  expect(
    roster.distribution.every((row: { count: number }) => row.count > 0),
  ).toBe(true);
  await expect(page.getByText("12 marks", { exact: true })).toBeVisible();
  expect(
    new Set(
      roster.students
        .filter((row: { diagnostic: unknown }) => row.diagnostic)
        .map((row: { diagnostic: { score: number } }) => row.diagnostic.score),
    ).size,
  ).toBeGreaterThan(4);

  const papers = await (await page.request.get("/api/teacher/papers")).json();
  const nextPaper = papers.find(
    (row: { title: string }) => row.title === "Next unit check",
  );
  const results = await (
    await page.request.get(`/api/teacher/papers/${nextPaper._id}/results`)
  ).json();
  expect(
    results.submissions.filter(
      (row: { status: string }) => row.status === "submitted",
    ),
  ).toHaveLength(3);
  expect(
    results.submissions.filter(
      (row: { status: string }) => row.status === "not_started",
    ),
  ).toHaveLength(21);

  const learnerContext = await browser.newContext();
  try {
    const learner = await learnerContext.newPage();
    learner.on("pageerror", (error) => errors.push(error.message));
    for (const sample of [
      {
        handle: "student",
        recommendation: "Fractions",
        session: "Fractions practice",
        score: "6 / 12",
      },
      {
        handle: "rohan",
        recommendation: "Linear equations",
        session: "Mixed practice",
        score: "9 / 12",
      },
    ]) {
      await learnerContext.clearCookies();
      await signIn(learner, sample.handle);
      await learner.getByRole("button", { name: "Learn", exact: true }).click();
      const recommendation = learner.locator(".p-hero-action");
      await expect(
        recommendation.getByRole("heading", { name: sample.recommendation }),
      ).toBeVisible();
      await learner.screenshot({
        path: `../.platform-runtime/demo-${sample.handle}-learn.png`,
        fullPage: true,
      });
      await learner
        .getByRole("button", { name: "Progress", exact: true })
        .click();
      await expect(
        learner.getByText(sample.score, { exact: true }),
      ).toBeVisible();
      await learner
        .getByRole("button", { name: "Session history", exact: true })
        .click();
      await learner
        .getByRole("link", { name: new RegExp(`^${sample.session} `) })
        .click();
      await expect(
        learner.getByRole("heading", { name: "Practice complete" }),
      ).toBeVisible();
      await learner
        .getByRole("button", { name: "Review answers", exact: true })
        .click();
      const answer = learner.locator(".p-review-list details").first();
      await answer.locator("summary").click();
      await expect(answer.getByText(/Correct answer:/)).toBeVisible();
      await expect(answer).not.toContainText("no stored explanation");
    }
    expect(errors).toEqual([]);
    await learnerContext.clearCookies();
    await signIn(learner, "admin");
    await learner
      .getByRole("button", { name: "Interactions", exact: true })
      .click();
    await expect(
      learner.getByRole("heading", { name: "Learning interactions" }),
    ).toBeVisible();
    const data = await (
      await learner.request.get("/api/admin/interactions")
    ).json();
    expect(data).toMatchObject({
      students: 24,
      sessions: 54,
      attempts: 561,
      invalid: 0,
      demo: true,
    });
    expect(data.correct + data.incorrect).toBe(data.attempts);
    for (const row of data.rows) {
      expect(row.student_id).toMatch(/^p-[0-9a-f]{24}$/);
      expect(row.source).toBe("synthetic");
    }
    const csv = await learner.request.post(
      "/api/admin/interactions/export?days=30",
      {
        headers: { Origin: "http://localhost:3000" },
      },
    );
    expect(csv.ok()).toBeTruthy();
    const exported = await csv.text();
    expect(exported.trimEnd().split(/\r?\n/)).toHaveLength(562);
    expect(exported).not.toMatch(/Asha|Rohan|@demo/);
  } finally {
    await learnerContext.close();
  }
});
