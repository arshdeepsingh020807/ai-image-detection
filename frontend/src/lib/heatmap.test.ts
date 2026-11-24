// @vitest-environment node
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { HEATMAP_MAX_ALPHA, HEATMAP_STOPS, heatmapGradient, stopAlpha } from "./heatmap";

const BACKEND = new URL("../../../backend/app/services/heatmap.py", import.meta.url);

describe("heatmap legend colour map", () => {
  it("reproduces the backend's inferno stops with opacity rising to 200/255", () => {
    expect(heatmapGradient()).toBe(
      "linear-gradient(90deg, rgba(120, 28, 109, 0) 0%, rgba(212, 72, 66, 0.261) 33.33%, rgba(250, 153, 8, 0.523) 66.67%, rgba(252, 255, 164, 0.784) 100%)",
    );
  });

  it("is transparent at no influence and most opaque at full influence", () => {
    expect(stopAlpha(0)).toBe(0);
    expect(stopAlpha(1)).toBe(0.784);
    expect(HEATMAP_STOPS.map((s) => stopAlpha(s.at))).toEqual([0, 0.261, 0.523, 0.784]);
  });

  it.skipIf(!existsSync(BACKEND))("matches the colours and alpha in backend/app/services/heatmap.py", () => {
    const source = readFileSync(BACKEND, "utf8");
    expect(source).toContain(`MAX_ALPHA = ${HEATMAP_MAX_ALPHA}`);
    const colours = /_COLOURS = np\.array\(\[(.*?)\], dtype/s.exec(source)?.[1] ?? "";
    expect([...colours.matchAll(/\[(\d+), (\d+), (\d+)\]/g)].map((m) => m.slice(1).map(Number))).toEqual(HEATMAP_STOPS.map((s) => [...s.rgb]));
    expect(source).toContain("_COLOUR_STOPS = np.array([0.0, 1 / 3, 2 / 3, 1.0])");
  });
});
