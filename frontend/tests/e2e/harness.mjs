import { spawn } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const require = createRequire(import.meta.url);
const root = fileURLToPath(new URL("../..", import.meta.url));
const fixture = (name) => readFileSync(new URL(`../fixtures/${name}`, import.meta.url), "utf8");

export const PORT = 4179;
export const BASE_URL = `http://127.0.0.1:${PORT}`;
export const CHROME = process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
export const SCREEN_DIR = process.env.SCREEN_DIR ?? "/private/tmp/claude-501/-Users-arshdeepsingh/0714b13b-f666-4746-9a0e-4f6fd3081f14/scratchpad/screens";
export const imagePath = (name) => fileURLToPath(new URL(`../fixtures/images/${name}`, import.meta.url));
export const VIEWPORTS = { phone: { width: 375, height: 812 }, tablet: { width: 768, height: 1024 }, desktop: { width: 1280, height: 900 } };

export const json = (status, body) => ({ status, contentType: "application/json", body: typeof body === "string" ? body : JSON.stringify(body) });
export const apiError = (status, code) => json(status, { error: { code, message: `${code} (mocked)` } });

export const DEFAULT_API = {
  "GET /api/health": () => json(200, fixture("health.json")),
  "GET /api/model": () => json(200, fixture("model.json")),
  "GET /api/metrics": () => json(200, fixture("metrics-full.json")),
  "POST /api/predict": () => json(200, fixture("prediction-ai.json")),
};
export const fixtures = {
  prediction: (name) => fixture(`prediction-${name}.json`),
  metricsEmpty: () => fixture("metrics-empty.json"),
};

export async function startPreview() {
  const vite = `${root}node_modules/vite/bin/vite.js`;
  const server = spawn(process.execPath, [vite, "preview", "--port", String(PORT), "--strictPort", "--host", "127.0.0.1"], { cwd: root, stdio: "ignore" });
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      if ((await fetch(BASE_URL)).ok) return server;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  server.kill();
  throw new Error("vite preview did not start");
}

export const launch = () => puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--no-first-run", "--no-default-browser-check"] });

export async function openPage(browser, { api = {}, mock = true, theme = "light", viewport = VIEWPORTS.desktop, reducedMotion = true } = {}) {
  const context = await browser.createBrowserContext();
  const page = await context.newPage();
  const errors = [];
  const requests = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  page.on("pageerror", (error) => { errors.push(error.message); });
  await page.setViewport(viewport);
  await page.emulateMediaFeatures([
    { name: "prefers-color-scheme", value: theme },
    { name: "prefers-reduced-motion", value: reducedMotion ? "reduce" : "no-preference" },
  ]);
  if (!mock) return { page, errors, requests };
  await page.setRequestInterception(true);
  const routes = { ...DEFAULT_API, ...api };
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (!url.pathname.startsWith("/api/")) return void request.continue();
    const key = `${request.method()} ${url.pathname}`;
    requests.push({ key, contentType: request.headers()["content-type"] ?? "" });
    const handler = routes[key];
    if (handler === "abort") return void request.abort("failed");
    if (handler === "hang") return;
    return void request.respond(handler ? handler(request) : apiError(404, "not_found"));
  });
  return { page, errors, requests };
}

const axeSource = readFileSync(require.resolve("axe-core/axe.min.js"), "utf8");

export async function runAxe(page) {
  await page.evaluate(axeSource);
  return page.evaluate(async () => {
    const result = await axe.run(document, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"] } });
    return result.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, nodes: v.nodes.map((n) => n.target.join(" ")).slice(0, 5) }));
  });
}

export const horizontalOverflow = (page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

export async function screenshot(page, name) {
  mkdirSync(SCREEN_DIR, { recursive: true });
  await page.evaluate(() => { window.scrollTo(0, 0); });
  await page.screenshot({ path: `${SCREEN_DIR}/${name}.png`, fullPage: true });
}

export async function uploadAndAnalyse(page, image = "landscape.jpg") {
  const input = await page.waitForSelector("#image-input");
  await input.uploadFile(imagePath(image));
  await page.waitForSelector(".stage[data-loaded]");
  await clickButton(page, "Analyse image");
  await page.waitForSelector(".verdict__title");
}

export async function clickButton(page, text) {
  const button = await page.waitForSelector(`::-p-xpath(//button[normalize-space()="${text}"])`);
  await button.click();
}
