import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HEATMAP_STOPS, stopAlpha } from "../lib/heatmap";
import { HeatmapControls } from "./HeatmapControls";

describe("HeatmapControls", () => {
  it("draws the legend from the shared colour map with end labels over a neutral backdrop", () => {
    const { container } = render(<HeatmapControls note="Server note." visible opacity={0.6} onVisibleChange={vi.fn()} onOpacityChange={vi.fn()} />);
    const ramp = container.querySelector<HTMLElement>(".heatmap-legend__ramp");
    const stops = [...(ramp?.style.backgroundImage ?? "").matchAll(/rgba\((\d+), (\d+), (\d+), ([\d.]+)\) ([\d.]+)%/g)];
    expect(stops.map((m) => [m[1], m[2], m[3]].map(Number))).toEqual(HEATMAP_STOPS.map((stop) => [...stop.rgb]));
    expect(stops.map((m) => Number(m[5]))).toEqual([0, 33.33, 66.67, 100]);
    stops.forEach((m, i) => { expect(Math.abs(Number(m[4]) - stopAlpha(HEATMAP_STOPS[i]?.at ?? 0))).toBeLessThanOrEqual(1 / 255); });
    expect(ramp?.parentElement).toHaveClass("heatmap-legend__bar");
    expect(screen.getByText("Less")).toBeInTheDocument();
    expect(screen.getByText("More influence")).toBeInTheDocument();
    expect(screen.getByText("Server note. The outline marks the region the model analysed.")).toBeInTheDocument();
    expect(document.querySelectorAll(".heatmap-controls p")).toHaveLength(1);
  });
});
