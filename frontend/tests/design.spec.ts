import { test, expect, type Page } from "@playwright/test";

const host = "demo.lumen.example.com";
export async function signIn(page: Page, role: string) {
  await page.goto("/login");
  await page.getByLabel("Email address").fill(`${role}@${host}`);
  await page.getByLabel("Password", { exact: true }).fill("Lumen12345");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText("Synthetic demo", { exact: true })).toBeVisible();
}

test("the reference layout works across desktop, tablet, and mobile without external fonts", async ({
  page,
}) => {
  const fontUrls: string[] = [];
  const errors: string[] = [];
  page.on("request", (r) => {
    if (/fonts\.(googleapis|gstatic)\.com/.test(r.url()))
      fontUrls.push(r.url());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await signIn(page, "student");
  await expect(
    page.getByRole("heading", { name: "Good to see you, Asha" }),
  ).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  const actual = await page
    .locator(".p-sidebar nav button")
    .first()
    .evaluate((x) => parseFloat(getComputedStyle(x).fontSize));
  expect(actual).toBeGreaterThanOrEqual(15);
  await page.screenshot({
    path: "../.platform-runtime/lumen-student-desktop.png",
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
  await page.getByRole("button", { name: "More", exact: true }).click();
  await page.getByRole("button", { name: "Open profile", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Learning profile" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../.platform-runtime/lumen-mobile-profile.png",
    fullPage: true,
  });
  expect(fontUrls).toEqual([]);
  expect(errors).toEqual([]);
});
