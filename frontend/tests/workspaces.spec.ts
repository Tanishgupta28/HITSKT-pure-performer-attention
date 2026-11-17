import { test, expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function signIn(page: Page, role: string) {
  await page.goto("/login");
  await page.getByLabel("Email address").fill(`${role}@demo.lumen.example.com`);
  await page.getByLabel("Password", { exact: true }).fill("Lumen12345");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText("Synthetic demo", { exact: true })).toBeVisible();
}

test("teacher assigns a fixed paper, student submits, teacher releases results", async ({
  page,
  browser,
}) => {
  await signIn(page, "teacher");
  await expect(
    page.getByRole("heading", { name: "Topics to revisit" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/lumen-teacher-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("navigation", { name: "Workspace" })
    .getByRole("button", { name: "Assessments" })
    .click();
  await page.getByRole("button", { name: "New assessment" }).click();
  const name = `QA unit paper ${Date.now()}`;
  await page.getByLabel("Assessment title").fill(name);
  await page
    .getByLabel("Class", { exact: true })
    .selectOption({ label: "8B Mathematics" });
  await page.getByLabel("Search questions").fill("FRA-018");
  await page.getByRole("checkbox", { name: /FRA-018/ }).check();
  await page.getByRole("button", { name: "Next: set rules" }).click();
  await page.getByRole("button", { name: "Review assignment" }).click();
  await page.getByRole("button", { name: "Assign to class" }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Assign to class" })
    .click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  const context = await browser.newContext(),
    student = await context.newPage();
  try {
    await signIn(student, "student");
    await student
      .getByRole("navigation", { name: "Workspace" })
      .getByRole("button", { name: "Assessments" })
      .click();
    await student
      .getByRole("row")
      .filter({ hasText: name })
      .getByRole("button", { name: "View instructions" })
      .click();
    await student
      .getByRole("button", { name: "Begin timed assessment" })
      .click();
    await expect(
      student.getByRole("heading", { name: "What is 1/2 + 1/3?" }),
    ).toBeVisible();
    await student.getByRole("radio", { name: /5\/6/ }).check();
    await student.getByRole("button", { name: "Save & review" }).click();
    await student
      .getByRole("button", { name: "Submit assessment", exact: true })
      .click();
    await expect(
      student.getByText("Waiting for your teacher to release results"),
    ).toBeVisible();
    await page.getByRole("button", { name: "Close paper" }).click();
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "Close assessment" })
      .click();
    await page
      .getByRole("button", { name: "Release results", exact: true })
      .click();
    await page
      .getByRole("dialog")
      .getByRole("button", { name: "Release results", exact: true })
      .click();
    await expect(
      page.getByText("Results released", { exact: true }),
    ).toBeVisible();
    await student.getByRole("button", { name: "Check status" }).click();
    await expect(
      student.getByRole("heading", { name: `${name} · report` }),
    ).toBeVisible();
  } finally {
    await context.close();
  }
});

test("admin publishes mapped content, exports data, and replays the two model paths", async ({
  page,
}) => {
  await signIn(page, "admin");
  await expect(
    page.getByRole("heading", { name: "Question bank" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/lumen-admin-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Add question" }).click();
  const id = `QA-NUM-${Date.now()}`;
  await page.getByLabel("Unique question ID").fill(id);
  await page.getByLabel("Topic", { exact: true }).selectOption("fractions");
  await page.getByLabel("Skill mapping").selectOption("fractions-foundations");
  await page.getByLabel("Question type").selectOption("numeric");
  await page
    .getByLabel("Question prompt")
    .fill("What is one half plus one half?");
  await page.getByLabel("Correct numeric answer").fill("1");
  await page
    .getByLabel("Explanation", { exact: true })
    .fill("Two halves make one whole.");
  await page.getByRole("button", { name: "Save and publish" }).click();
  await expect(
    page.getByRole("heading", { name: "Question bank" }),
  ).toBeVisible();
  await page.getByLabel("Search questions").fill(id);
  await expect(page.getByRole("row").filter({ hasText: id })).toContainText(
    "Published",
  );

  await page.getByRole("button", { name: "Interactions", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Learning interactions" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Review export" }).click();
  const transfer = page.waitForEvent("download");
  await page.getByRole("button", { name: "Confirm export" }).click();
  const csv = await readFile((await (await transfer).path())!, "utf-8");
  expect(csv).toContain("student_id,session_id,question_id");
  expect(csv).toContain("synthetic");
  expect(csv).not.toContain("@demo");

  await page.getByRole("button", { name: "Model Lab", exact: true }).click();
  await page.getByRole("button", { name: "Try BKT replay" }).click();
  await expect(
    page.getByText("Hidden before reveal", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Run prediction" }).click();
  await expect(
    page.getByText("Computed now using the displayed BKT history."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Reveal target" }).click();
  await expect(
    page.getByText("Correct · synthetic target label", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Model", { exact: true }).selectOption("hitskt");
  await page.getByRole("button", { name: "Run prediction" }).click();
  await expect(page.getByText("38.64%", { exact: true })).toBeVisible({
    timeout: 60_000,
  });
  await page.screenshot({
    path: "../.platform-runtime/lumen-model-lab.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Independent HiTSKT replay" }),
  ).toBeVisible();
});
