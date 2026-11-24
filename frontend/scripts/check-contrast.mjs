import { readFileSync } from "node:fs";

const TEXT = 4.5;
const NON_TEXT = 3;

export const PAIRS = [
  ["text", "bg", TEXT],
  ["text", "surface", TEXT],
  ["text", "surface-sunken", TEXT],
  ["text-muted", "bg", TEXT],
  ["text-muted", "surface", TEXT],
  ["text-muted", "surface-sunken", TEXT],
  ["text-subtle", "bg", TEXT],
  ["text-subtle", "surface", TEXT],
  ["text-subtle", "surface-sunken", TEXT],
  ["accent", "bg", TEXT],
  ["accent", "surface", TEXT],
  ["accent", "accent-soft", TEXT],
  ["accent-hover", "surface", TEXT],
  ["on-accent", "accent", TEXT],
  ["on-accent", "accent-hover", TEXT],
  ["text", "accent-soft", TEXT],
  ["text", "real-soft", TEXT],
  ["text", "ai-soft", TEXT],
  ["text", "inconclusive-soft", TEXT],
  ["text-muted", "real-soft", TEXT],
  ["text-muted", "ai-soft", TEXT],
  ["text-muted", "inconclusive-soft", TEXT],
  ["danger", "surface", TEXT],
  ["danger", "danger-soft", TEXT],
  ["text", "danger-soft", TEXT],
  ["warning", "warning-soft", TEXT],
  ["text", "warning-soft", TEXT],
  ["border-strong", "surface", NON_TEXT],
  ["border-strong", "bg", NON_TEXT],
  ["border-strong", "surface-sunken", NON_TEXT],
  ["focus", "surface", NON_TEXT],
  ["focus", "bg", NON_TEXT],
  ["real", "surface", NON_TEXT],
  ["ai", "surface", NON_TEXT],
  ["inconclusive", "surface", NON_TEXT],
  ["real", "surface-sunken", NON_TEXT],
  ["ai", "surface-sunken", NON_TEXT],
];

export function parseBlocks(css) {
  const block = (selector) => {
    const start = css.indexOf(selector);
    if (start < 0) throw new Error(`Missing block: ${selector}`);
    const open = css.indexOf("{", start);
    const body = css.slice(open + 1, css.indexOf("}", open));
    return Object.fromEntries([...body.matchAll(/--color-([\w-]+):\s*(#[0-9a-f]{6})\s*;/gi)].map((m) => [m[1], m[2]]));
  };
  return {
    light: block(":root {"),
    darkMedia: block(':root:not([data-theme="light"])'),
    dark: block(':root[data-theme="dark"]'),
  };
}

function luminance(hex) {
  const channel = (i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5);
}

export function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

export function audit(themes) {
  const failures = [];
  const rows = [];
  for (const [theme, colors] of Object.entries(themes)) {
    for (const [fg, bg, min] of PAIRS) {
      if (!colors[fg] || !colors[bg]) {
        failures.push(`${theme}: missing --color-${colors[fg] ? bg : fg}`);
        continue;
      }
      const ratio = contrast(colors[fg], colors[bg]);
      rows.push({ theme, pair: `${fg} on ${bg}`, ratio: ratio.toFixed(2), min, ok: ratio >= min });
      if (ratio < min) failures.push(`${theme}: ${fg} on ${bg} is ${ratio.toFixed(2)}:1 (needs ${min}:1)`);
    }
  }
  return { rows, failures };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { light, dark, darkMedia } = parseBlocks(readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8"));
  const drift = Object.keys({ ...dark, ...darkMedia }).filter((key) => dark[key] !== darkMedia[key]);
  const { rows, failures } = audit({ light, dark });
  console.table(rows);
  if (drift.length) failures.push(`Dark theme blocks disagree on: ${drift.join(", ")}`);
  if (failures.length) {
    console.error(`\nWCAG AA contrast check failed:\n  ${failures.join("\n  ")}`);
    process.exit(1);
  }
  console.log(`\nAll ${rows.length} token pairs meet WCAG AA in both themes.`);
}
