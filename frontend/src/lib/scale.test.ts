import { describe, expect, it } from "vitest";
import { groupedBar, linearScale, linePath, nearestIndex, niceDomain, niceStep, ticks } from "./scale";

describe("linearScale", () => {
  it("maps and inverts", () => {
    const y = linearScale([0, 1], [212, 12]);
    expect(y(0)).toBe(212);
    expect(y(1)).toBe(12);
    expect(y(0.25)).toBe(162);
    expect(y.invert(162)).toBe(0.25);
  });

  it("handles an offset domain", () => {
    const x = linearScale([1, 10], [56, 506]);
    expect(x(1)).toBe(56);
    expect(x(4)).toBe(206); // 56 + 3 * 50
    expect(x.invert(456)).toBe(9);
  });

  it("collapses a zero-width domain to the range start", () => {
    const x = linearScale([5, 5], [0, 100]);
    expect(x(5)).toBe(0);
    expect(x.invert(40)).toBe(5);
  });
});

describe("ticks and nice domains", () => {
  it("picks the nearest 1-2-5 step", () => {
    expect(niceStep(1, 5)).toBe(0.2); // raw 0.2 -> ratio 2 -> 2
    expect(niceStep(6100, 4)).toBe(2000); // raw 1525 -> ratio 1.525 >= sqrt(2) -> 2
    expect(niceStep(0.52, 4)).toBeCloseTo(0.1); // raw 0.13 -> ratio 1.3 < sqrt(2) -> 1
    expect(niceStep(0.94, 4)).toBeCloseTo(0.2); // raw 0.235 -> ratio 2.35 -> 2
    expect(niceStep(400, 1)).toBe(500); // raw 400 -> ratio 4 >= sqrt(10) -> 5
    expect(niceStep(9, 1)).toBe(10); // raw 9 -> ratio 9 >= sqrt(50) -> 10
  });

  it("produces clean tick values without float noise", () => {
    expect(ticks([0, 1], 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(ticks([0, 8000], 4)).toEqual([0, 2000, 4000, 6000, 8000]);
    expect(ticks([1, 10], 10)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(ticks([0.03, 0.97], 4)).toEqual([0.2, 0.4, 0.6, 0.8]);
    expect(ticks([3, 3])).toEqual([3]);
  });

  it("rounds domains outwards to the step", () => {
    expect(niceDomain(0, 6100, 4)).toEqual([0, 8000]);
    expect(niceDomain(0.06, 0.52, 4)).toEqual([0, 0.6]);
    expect(niceDomain(2, 2)).toEqual([2, 3]);
  });
});

describe("linePath", () => {
  it("builds an SVG path in pixel space", () => {
    const x = linearScale([0, 1], [0, 200]);
    const y = linearScale([0, 1], [100, 0]);
    expect(linePath([[0, 0], [0.5, 0.25], [1, 1]], x, y)).toBe("M0,100L100,75L200,0");
  });

  it("rounds to two decimals", () => {
    const x = linearScale([0, 3], [0, 100]);
    expect(linePath([[1, 0], [2, 0]], x, x)).toBe("M33.33,0L66.67,0");
  });
});

describe("nearestIndex", () => {
  it("finds the closest value in unsorted data", () => {
    expect(nearestIndex([1, 0.8, 0.3, 0], 0.35)).toBe(2);
    expect(nearestIndex([0, 0.1, 0.5], 0.9)).toBe(2);
    expect(nearestIndex([], 1)).toBe(0);
  });
});

describe("groupedBar", () => {
  it("splits a bin into equal slots separated by gaps", () => {
    // bin 100..160 (60px), 2 slots, gap 2: width = (60 - 6) / 2 = 27
    expect(groupedBar(100, 160, 0, 2, 2)).toEqual({ x: 102, width: 27 });
    expect(groupedBar(100, 160, 1, 2, 2)).toEqual({ x: 131, width: 27 });
  });

  it("never returns a negative width", () => {
    expect(groupedBar(0, 2, 0, 2, 2).width).toBe(0);
  });
});
