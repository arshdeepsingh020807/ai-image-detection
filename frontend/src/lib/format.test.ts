import { describe, expect, it } from "vitest";
import { formatBytes, formatCount, formatDate, formatDecimal, formatMs, formatPercent, formatProbability, humanize } from "./format";

describe("format", () => {
  it("formats percentages and missing values", () => {
    expect(formatPercent(0.9312)).toBe("93.1%");
    expect(formatPercent(1, 0)).toBe("100%");
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
  });

  it("never shows a probability as certain", () => {
    expect(formatProbability(0.99976)).toBe(">99.9%");
    expect(formatProbability(0.9995)).toBe(">99.9%");
    expect(formatProbability(0.9994)).toBe("99.9%");
    expect(formatProbability(0.00024)).toBe("<0.1%");
    expect(formatProbability(0)).toBe("<0.1%");
    expect(formatProbability(0.0005)).toBe("<0.1%");
    expect(formatProbability(0.0006)).toBe("0.1%");
    expect(formatProbability(0.5)).toBe("50.0%");
  });

  it("formats decimals", () => {
    expect(formatDecimal(0.98691)).toBe("0.987");
    expect(formatDecimal(null)).toBe("—");
  });

  it("formats counts with separators", () => {
    expect(formatCount(127161)).toBe("127,161");
  });

  it("formats bytes with binary units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(245760)).toBe("240.0 KB");
    expect(formatBytes(10 * 1024 * 1024)).toBe("10.0 MB");
  });

  it("formats milliseconds", () => {
    expect(formatMs(4.25)).toBe("4.3 ms");
    expect(formatMs(1234.4)).toBe("1,234 ms");
  });

  it("humanizes keys", () => {
    expect(humanize("external_leak")).toBe("External leak");
    expect(humanize("DALL-E")).toBe("DALL-E");
  });

  it("formats ISO dates in UTC and passes through unparseable text", () => {
    expect(formatDate("2026-11-20T10:30:00Z")).toBe("20 Nov 2026, 10:30 UTC");
    expect(formatDate("yesterday")).toBe("yesterday");
  });
});
