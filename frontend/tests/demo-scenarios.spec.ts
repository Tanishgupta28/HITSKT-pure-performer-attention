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

test("demo learners have distinct practice histories and admins can export interactions", async ({
  browser,
}) => {
  const errors: string[] = [];
  const learnerContext = await browser.newContext();
  try {
    const learner = await learnerContext.newPage();
    learner.on("pageerror", (error) => errors.push(error.message));
    for (const sample of [
      {
        handle: "student",
        session: "Fractions practice",
        score: "3 / 6",
      },
      {
        handle: "rohan",
        session: "Mixed practice",
        score: "9 / 12",
      },
    ]) {
      await learnerContext.clearCookies();
      await signIn(learner, sample.handle);
      await learner.getByRole("button", { name: "Learn", exact: true }).click();
      const recommendation = learner.locator(".p-hero-action");
      await expect(recommendation.getByRole("heading")).toBeVisible();
      await learner.screenshot({
        path: `../.platform-runtime/demo-${sample.handle}-learn.png`,
        fullPage: true,
      });
      await learner
        .getByRole("button", { name: "Progress", exact: true })
        .click();
      await learner
        .getByRole("button", { name: "Session history", exact: true })
        .click();
      await expect(
        learner.getByText(sample.score, { exact: true }).first(),
      ).toBeVisible();
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
      sessions: 21,
      attempts: 168,
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
    expect(exported.trimEnd().split(/\r?\n/)).toHaveLength(169);
    expect(exported).not.toMatch(/Asha|Rohan|@demo/);
  } finally {
    await learnerContext.close();
  }
});
