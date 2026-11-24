import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { after, before, describe, it } from "node:test";
import {
  apiError, BASE_URL, clickButton, fixtures, horizontalOverflow, imagePath, json, launch, openPage, runAxe, screenshot,
  startPreview, uploadAndAnalyse, VIEWPORTS,
} from "./harness.mjs";

let server;
let browser;

before(async () => {
  server = await startPreview();
  browser = await launch();
});

after(async () => {
  await browser?.close();
  server?.kill();
});

async function visit(options, path = "/") {
  const session = await openPage(browser, options);
  await session.page.goto(`${BASE_URL}${path}`, { waitUntil: "networkidle0" });
  return session;
}

const textOf = (page, selector) => page.$eval(selector, (el) => el.textContent.trim());
const activeText = (page) => page.evaluate(() => document.activeElement?.textContent?.trim() ?? "");

describe("detect flow", () => {
  it("uploads through the real file input and shows the verdict with a correctly placed heatmap", async () => {
    const { page, errors, requests } = await visit({});
    await uploadAndAnalyse(page);
    assert.equal(await textOf(page, ".verdict__title"), "Likely AI-generated");
    assert.equal(await activeText(page), "Likely AI-generated");
    const predict = requests.find((r) => r.key === "POST /api/predict");
    assert.match(predict.contentType, /^multipart\/form-data; boundary=/);

    const geometry = await page.evaluate(() => {
      const image = document.querySelector(".stage__image").getBoundingClientRect();
      const region = document.querySelector(".stage__overlay").getBoundingClientRect();
      const heatmap = document.querySelector(".stage__heatmap");
      return { image: image.toJSON(), region: region.toJSON(), heatmapLoaded: heatmap.complete && heatmap.naturalWidth === 224 };
    });
    assert.ok(geometry.heatmapLoaded, "heatmap PNG decoded");
    const { image, region } = geometry;
    const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1, `${actual} ≈ ${expected}`);
    close(region.left - image.left, 0.28125 * image.width);
    close(region.top - image.top, 0.0625 * image.height);
    close(region.width, 0.4375 * image.width);
    close(region.height, 0.875 * image.height);
    close(image.width / image.height, 2);
    assert.deepEqual(errors, []);
  });

  it("accepts a drag-and-drop upload", async () => {
    const { page, errors } = await visit({ api: { "POST /api/predict": () => json(200, fixtures.prediction("real")) } });
    const bytes = readFileSync(imagePath("portrait.png")).toString("base64");
    await page.evaluate((b64) => {
      const data = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      const transfer = new DataTransfer();
      transfer.items.add(new File([data], "portrait.png", { type: "image/png" }));
      const zone = document.querySelector(".dropzone");
      zone.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, dataTransfer: transfer }));
      zone.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: transfer }));
    }, bytes);
    await page.waitForSelector(".stage[data-loaded]");
    await clickButton(page, "Analyse image");
    await page.waitForSelector(".verdict__title");
    assert.equal(await textOf(page, ".verdict__title"), "Likely real");
    assert.equal(await page.$(".stage__overlay"), null);
    assert.deepEqual(errors, []);
  });

  it("can be completed with the keyboard alone", async () => {
    const { page, errors } = await visit({});
    await page.keyboard.press("Tab");
    assert.equal(await activeText(page), "Skip to main content");
    await page.keyboard.press("Enter");
    assert.equal(await page.evaluate(() => document.activeElement?.id), "main");
    let id = "";
    for (let i = 0; i < 10 && id !== "image-input"; i++) {
      await page.keyboard.press("Tab");
      id = await page.evaluate(() => document.activeElement?.id);
    }
    assert.equal(id, "image-input");
    const outline = await page.$eval(".dropzone", (el) => getComputedStyle(el).outlineStyle);
    assert.equal(outline, "solid", "drop zone shows a visible focus ring");
    const [chooser] = await Promise.all([page.waitForFileChooser(), page.keyboard.press("Space")]);
    await chooser.accept([imagePath("photo.webp")]);
    await page.waitForSelector(".stage[data-loaded]");
    let label = "";
    for (let i = 0; i < 6 && label !== "Analyse image"; i++) {
      await page.keyboard.press("Tab");
      label = await activeText(page);
    }
    assert.equal(label, "Analyse image");
    await page.keyboard.press("Enter");
    await page.waitForSelector(".verdict__title");
    assert.equal(await activeText(page), "Likely AI-generated");
    assert.deepEqual(errors, []);
  });

  it("rejects a renamed text file and a GIF in the browser without uploading", async () => {
    const { page, requests } = await visit({});
    const input = await page.$("#image-input");
    await input.uploadFile(imagePath("not-an-image.jpg"));
    await page.waitForSelector("[role=alert]");
    assert.equal(await textOf(page, "[role=alert] h3"), "This file could not be read as an image");
    await input.uploadFile(imagePath("still.gif"));
    await page.waitForFunction(() => document.querySelector("[role=alert] h3")?.textContent === "Unsupported file type");
    assert.equal(requests.filter((r) => r.key === "POST /api/predict").length, 0);
  });

  it("explains an inconclusive result and warns about an untrained model", async () => {
    const { page } = await visit({ api: { "POST /api/predict": () => json(200, fixtures.prediction("untrained")) } });
    await uploadAndAnalyse(page);
    assert.match(await textOf(page, ".notice--warning"), /untrained stand-in/);
    assert.deepEqual(await runAxe(page), []);
    const second = await visit({ api: { "POST /api/predict": () => json(200, fixtures.prediction("inconclusive")) } });
    await uploadAndAnalyse(second.page);
    assert.match(await textOf(second.page, ".result__summary"), /inside the uncertainty band/);
  });

  it("maps API errors and recovers with retry", async () => {
    let calls = 0;
    const { page, errors } = await visit({
      api: { "POST /api/predict": () => (++calls === 1 ? apiError(429, "rate_limited") : json(200, fixtures.prediction("ai"))) },
    });
    const input = await page.waitForSelector("#image-input");
    await input.uploadFile(imagePath("landscape.jpg"));
    await page.waitForSelector(".stage[data-loaded]");
    await clickButton(page, "Analyse image");
    await page.waitForSelector("[role=alert]");
    assert.equal(await activeText(page), "Too many requests");
    await clickButton(page, "Try again");
    await page.waitForSelector(".verdict__title");
    assert.equal(calls, 2);
    assert.deepEqual(errors.filter((e) => !e.includes("429")), []);
  });

  it("reports a network failure", async () => {
    const { page } = await visit({ api: { "POST /api/predict": "abort" } });
    const input = await page.waitForSelector("#image-input");
    await input.uploadFile(imagePath("landscape.jpg"));
    await page.waitForSelector(".stage[data-loaded]");
    await clickButton(page, "Analyse image");
    await page.waitForSelector("[role=alert]");
    assert.equal(await textOf(page, "[role=alert] h3"), "Could not reach the server");
  });

  it("does not shift layout when an image is selected and analysed", async () => {
    const { page } = await visit({});
    const zone = await page.$eval(".dropzone", (el) => el.getBoundingClientRect().toJSON());
    const resultTop = await page.$eval(".panel--result", (el) => el.getBoundingClientRect().top);
    await uploadAndAnalyse(page);
    const stage = await page.$eval(".stage", (el) => el.getBoundingClientRect().toJSON());
    assert.deepEqual([stage.width, stage.height, stage.top], [zone.width, zone.height, zone.top]);
    assert.equal(await page.$eval(".panel--result", (el) => el.getBoundingClientRect().top), resultTop);
    const cls = await page.evaluate(() => new Promise((resolve) => {
      new PerformanceObserver((list) => {
        resolve(list.getEntries().filter((e) => !e.hadRecentInput).reduce((sum, e) => sum + e.value, 0));
      }).observe({ type: "layout-shift", buffered: true });
      setTimeout(() => { resolve(0); }, 500);
    }));
    assert.ok(cls < 0.01, `cumulative layout shift ${cls}`);
  });
});

describe("navigation", () => {
  it("reaches every page from the nav bar and marks it current", async () => {
    const { page, errors } = await visit({});
    for (const [label, heading] of [["Performance", "Model performance"], ["Method", "Method"], ["Detect", "Detect AI-generated images"]]) {
      const link = await page.waitForSelector(`::-p-xpath(//nav[@aria-label="Main"]//a[normalize-space()="${label}"])`);
      await link.click();
      await page.waitForFunction((h) => document.querySelector("h1")?.textContent === h, {}, heading);
      assert.equal(await page.$eval('nav[aria-label="Main"] [aria-current="page"]', (el) => el.textContent), label);
    }
    assert.deepEqual(errors, []);
  });

  it("serves a 404 page for unknown routes", async () => {
    const { page, errors } = await visit({}, "/does/not/exist");
    assert.equal(await textOf(page, "h1"), "Page not found");
    assert.equal(await page.title(), "Page not found · AI Image Detection");
    assert.deepEqual(errors, []);
  });

  it("persists the theme toggle across reloads", async () => {
    const { page } = await visit({ theme: "light" });
    await page.click(".theme-toggle");
    assert.equal(await page.$eval("html", (el) => el.dataset.theme), "dark");
    await page.reload({ waitUntil: "networkidle0" });
    assert.equal(await page.$eval("html", (el) => el.dataset.theme), "dark");
    assert.equal(await page.$eval(".theme-toggle", (el) => el.getAttribute("aria-pressed")), "true");
    const background = await page.$eval("body", (el) => getComputedStyle(el).backgroundColor);
    assert.equal(background, "rgb(15, 18, 22)");
  });

  it("is served with a Content-Security-Policy that blocks inline script", async () => {
    const response = await fetch(BASE_URL);
    assert.match(response.headers.get("content-security-policy"), /script-src 'self'/);
    const html = await response.text();
    assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)[^>]*>/);
  });
});

describe("performance page", () => {
  it("renders every published section", async () => {
    const { page, errors } = await visit({}, "/performance");
    await page.waitForSelector("#headline .stat-tiles");
    assert.equal(await page.$$eval(".empty-state", (els) => els.length), 0);
    assert.equal(await page.$$eval(".report-section svg[role=img]", (els) => els.length), 5);
    assert.deepEqual(errors, []);
  });

  it("shows honest empty states when nothing is published", async () => {
    const { page, errors } = await visit({ api: { "GET /api/metrics": () => json(200, fixtures.metricsEmpty()) } }, "/performance");
    await page.waitForSelector(".empty-state");
    assert.equal(await page.$$eval(".empty-state", (els) => els.length), 11);
    assert.equal(await page.$$eval(".report-section svg[role=img]", (els) => els.length), 0);
    assert.deepEqual(await runAxe(page), []);
    assert.deepEqual(errors, []);
  });
});

const PAGES = [
  ["detect", "/", "h1"],
  ["performance", "/performance", "#headline .stat-tiles"],
  ["method", "/method", "h1"],
  ["404", "/missing-page", "h1"],
];

describe("responsive layout and accessibility", () => {
  for (const theme of ["light", "dark"]) {
    for (const [name, viewport] of Object.entries(VIEWPORTS)) {
      it(`${theme} ${name}: no horizontal overflow, no axe violations, no console errors`, async () => {
        const { page, errors } = await openPage(browser, { theme, viewport });
        for (const [pageName, path, ready] of PAGES) {
          await page.goto(`${BASE_URL}${path}`, { waitUntil: "networkidle0" });
          await page.waitForSelector(ready);
          assert.equal(await horizontalOverflow(page), 0, `${pageName} overflows horizontally`);
          assert.deepEqual(await runAxe(page), [], `${pageName} axe violations`);
          await screenshot(page, `${pageName}-${theme}-${name}`);
        }
        await page.goto(`${BASE_URL}/`, { waitUntil: "networkidle0" });
        await uploadAndAnalyse(page);
        assert.equal(await horizontalOverflow(page), 0, "result overflows horizontally");
        assert.deepEqual(await runAxe(page), [], "result axe violations");
        await screenshot(page, `result-${theme}-${name}`);
        assert.deepEqual(errors, []);
      });
    }
  }
});
