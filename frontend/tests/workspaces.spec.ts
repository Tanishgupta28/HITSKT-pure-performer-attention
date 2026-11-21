import { test, expect, type Page } from "@playwright/test";
import { readFile } from "node:fs/promises";

async function signIn(page: Page, role: string) {
  await page.goto("/login");
  await page.getByLabel("Email address").fill(`${role}@demo.lumen.example.com`);
  await page.getByLabel("Password", { exact: true }).fill("Lumen12345");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText("Synthetic demo", { exact: true })).toBeVisible();
}

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
    timeout: 120_000,
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
