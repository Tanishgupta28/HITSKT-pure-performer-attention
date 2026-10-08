import { expect, test } from "@playwright/test";

async function account(page: import("@playwright/test").Page) {
  const response = await page.request.post("/api/auth/register", {
    data: {
      name: "Design Preview",
      email: `browser-design-${Date.now()}-${Math.random().toString(16).slice(2)}@example.com`,
      password: "design-preview-learning-42",
    },
  });
  expect(response.status()).toBe(201);
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Hello, Design" }),
  ).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
}

test("readable local fonts, Lottie motion controls, and responsive layouts", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const externalFonts: string[] = [];
  page.on("request", (request) => {
    if (/fonts\.(googleapis|gstatic)\.com/.test(request.url()))
      externalFonts.push(request.url());
  });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Start something good." }),
  ).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: "../.platform-runtime/redesign-auth-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: "../.platform-runtime/redesign-auth-mobile.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1080 });
  await account(page);
  const scene = page.locator('[data-lottie="lumen-orbit"]');
  await expect(scene.locator("svg")).toBeVisible();
  await expect(scene).toHaveAttribute("data-playback", "playing");
  const typography = await page.evaluate(() => {
    const style = (selector: string) =>
      getComputedStyle(document.querySelector(selector)!);
    return {
      body: style("body").fontFamily,
      heading: style("h1").fontFamily,
      copy: parseFloat(style(".hero-copy p").fontSize),
      nav: parseFloat(style(".sidebar nav button").fontSize),
      icon: document
        .querySelector(".sidebar nav button svg")!
        .getBoundingClientRect().width,
      fontsLoaded:
        document.fonts.check(`400 16px ${style("body").fontFamily}`) &&
        document.fonts.check(`600 40px ${style("h1").fontFamily}`),
    };
  });
  expect(typography.copy).toBeGreaterThanOrEqual(16);
  expect(typography.nav).toBeGreaterThanOrEqual(16);
  expect(typography.icon).toBeGreaterThanOrEqual(23);
  expect(typography.body).toContain("bodyFont");
  expect(typography.heading).toContain("headingFont");
  expect(typography.fontsLoaded).toBe(true);
  await page.getByRole("button", { name: "Pause animations" }).click();
  await expect(scene).toHaveAttribute("data-playback", "paused");
  await expect(page.locator("html")).toHaveAttribute("data-motion", "paused");
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Resume animations" }),
  ).toBeVisible();
  await expect(page.locator('[data-lottie="lumen-orbit"]')).toHaveAttribute(
    "data-playback",
    "paused",
  );
  await page.screenshot({
    path: "../.platform-runtime/redesign-dashboard-desktop.png",
    fullPage: true,
  });
  for (const width of [1024, 768, 390, 360]) {
    await page.setViewportSize({ width, height: 900 });
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    if (width === 390)
      await page.screenshot({
        path: "../.platform-runtime/redesign-dashboard-mobile.png",
        fullPage: true,
      });
  }
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "My profile", exact: true }).click();
  await expect(page.getByLabel("Grade or learning stage")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../.platform-runtime/redesign-profile-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
  expect(externalFonts).toEqual([]);
});

test.describe("system reduced motion", () => {
  test.use({ reducedMotion: "reduce" });
  test("keeps the artwork visible without animation", async ({ page }) => {
    await account(page);
    const scene = page.locator('[data-lottie="lumen-orbit"]');
    await expect(scene.locator("svg")).toBeVisible();
    await expect(scene).toHaveAttribute("data-playback", "paused");
    await expect(
      page.getByRole("button", { name: "Pause animations" }),
    ).toHaveCount(0);
    expect(
      await page
        .locator(".brand-mark--sculpture")
        .evaluate((element) => getComputedStyle(element).animationName),
    ).toBe("none");
  });
});
