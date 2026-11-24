import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { clickButton, horizontalOverflow, imagePath, launch, openPage, runAxe, screenshot } from "./harness.mjs";

const BACKEND = process.env.BACKEND_URL;
let browser;

describe("integration with the real backend", { skip: BACKEND ? false : "set BACKEND_URL to run" }, () => {
  before(async () => { browser = await launch(); });
  after(async () => { await browser?.close(); });

  const visit = async (path, options = {}) => {
    const session = await openPage(browser, { mock: false, ...options });
    await session.page.goto(`${BACKEND}${path}`, { waitUntil: "networkidle0" });
    return session;
  };

  it("serves the built app under the backend's CSP", async () => {
    const response = await fetch(`${BACKEND}/performance`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get("content-security-policy") ?? "", /script-src 'self'/);
    assert.match(await response.text(), /<div id="root">/);
  });

  it("analyses a real upload with the untrained stand-in and shows the heatmap and warning", async () => {
    for (const theme of ["light", "dark"]) {
      const { page, errors } = await visit("/", { theme });
      const input = await page.waitForSelector("#image-input");
      await input.uploadFile(imagePath("landscape.jpg"));
      await page.waitForSelector(".stage[data-loaded]");
      await clickButton(page, "Analyse image");
      await page.waitForSelector(".verdict__title", { timeout: 30_000 });
      assert.match(await page.$eval(".notice--warning", (el) => el.textContent), /untrained stand-in/);
      assert.notEqual(await page.$eval(".verdict__title", (el) => el.textContent), "Inconclusive");
      assert.ok(await page.$eval(".stage__heatmap", (img) => img.complete && img.naturalWidth === 224));
      assert.deepEqual(await runAxe(page), []);
      assert.equal(await horizontalOverflow(page), 0);
      await screenshot(page, `integration-result-${theme}`);
      const legend = await page.$(".heatmap-controls");
      await legend.screenshot({ path: `${process.env.SCREEN_DIR ?? "/private/tmp/claude-501/-Users-arshdeepsingh/0714b13b-f666-4746-9a0e-4f6fd3081f14/scratchpad/screens"}/legend-${theme}.png` });
      assert.deepEqual(errors, []);
    }
  });

  it("rejects a renamed text file through the client check and a GIF via the server when forced", async () => {
    const { page, errors } = await visit("/");
    const input = await page.waitForSelector("#image-input");
    await input.uploadFile(imagePath("not-an-image.jpg"));
    await page.waitForSelector("[role=alert]");
    const status = await page.evaluate(async () => {
      const form = new FormData();
      form.append("file", new File([new Uint8Array([71, 73, 70, 56, 57, 97])], "x.gif"));
      form.append("explain", "maybe");
      const response = await fetch("/api/predict", { method: "POST", body: form });
      return [response.status, (await response.json()).error.code];
    });
    assert.deepEqual(status, [400, "invalid_request"]);
    assert.deepEqual(errors.filter((e) => !e.includes("400")), []);
  });

  it("renders the performance page from the real metrics without console errors", async () => {
    const { page, errors } = await visit("/performance");
    await page.waitForSelector("#dataset .stat-tiles");
    assert.ok((await page.$$eval(".empty-state", (els) => els.length)) > 0, "unpublished files show empty states");
    assert.match(await page.$eval("#model-card", (el) => el.textContent), /untrained stand-in/);
    assert.deepEqual(await runAxe(page), []);
    assert.deepEqual(errors, []);
  });

  it("falls back to index.html for client routes and shows the 404 page", async () => {
    const { page, errors } = await visit("/no/such/page");
    assert.equal(await page.$eval("h1", (el) => el.textContent), "Page not found");
    assert.match(await page.$eval(".service-status", (el) => el.textContent), /Service online/);
    assert.deepEqual(errors, []);
  });
});
