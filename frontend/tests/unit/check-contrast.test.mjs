// @vitest-environment node
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { audit, contrast, parseBlocks, PAIRS } from "../../scripts/check-contrast.mjs";

const tokens = readFileSync(new URL("../../src/styles/tokens.css", import.meta.url), "utf8");

describe("check-contrast script", () => {
  it("computes WCAG ratios", () => {
    expect(contrast("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrast("#777777", "#ffffff")).toBeCloseTo(4.48, 2);
    expect(contrast("#ffffff", "#ffffff")).toBe(1);
  });

  it("reads each theme from its own block", () => {
    const { light, dark, darkMedia } = parseBlocks(tokens);
    expect(light.bg).toBe("#f5f6f8");
    expect(dark.bg).toBe("#0f1216");
    expect(darkMedia).toEqual(dark);
  });

  it("passes the shipped tokens", () => {
    const { light, dark } = parseBlocks(tokens);
    const { failures, rows } = audit({ light, dark });
    expect(failures).toEqual([]);
    expect(rows).toHaveLength(PAIRS.length * 2);
  });

  it("fails a light text colour below 4.5:1 and reports missing tokens", () => {
    const { light, dark } = parseBlocks(tokens.replace("--color-text-subtle: #5d6674;", "--color-text-subtle: #8a93a0;"));
    expect(audit({ light, dark }).failures).toEqual(expect.arrayContaining([expect.stringMatching(/^light: text-subtle on surface is 3\.\d\d:1/)]));
    expect(audit({ light: { ...light, bg: undefined }, dark }).failures).toContain("light: missing --color-bg");
  });
});
