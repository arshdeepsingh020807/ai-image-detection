import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BarList } from "./BarList";
import { ConfusionMatrix } from "./ConfusionMatrix";
import { Histogram } from "./Histogram";
import { Legend } from "./Legend";
import { LineChart } from "./LineChart";

const fmt = (v: number) => v.toFixed(1);

describe("LineChart", () => {
  const renderRoc = () => render(
    <LineChart
      title="ROC" description="desc" square diagonal
      series={[{ id: "roc", label: "ROC", tone: "accent", points: [[0, 0], [0.5, 0.75], [1, 1]] }]}
      xDomain={[0, 1]} yDomain={[0, 1]} xLabel="FPR" yLabel="TPR" xFormat={fmt} yFormat={fmt}
    />,
  );

  it("draws the path in pixel space at the fallback width", () => {
    renderRoc();
    // width 560 -> plot 56..544 (488 wide); square height min(488, 320) = 320 -> y 332..12
    // (0.5, 0.75) -> x = 56 + 244 = 300, y = 332 - 240 = 92
    expect(document.querySelector("path.line")).toHaveAttribute("d", "M56,332L300,92L544,12");
    expect(document.querySelector("line.reference")).toHaveAttribute("x2", "544");
    expect(screen.getByRole("img", { name: "ROC" })).toHaveAccessibleDescription("desc");
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });

  it("shows a crosshair tooltip at the nearest point and hides it on leave", () => {
    renderRoc();
    const hit = document.querySelector(".hit-area") as SVGRectElement;
    hit.getBoundingClientRect = () => ({ left: 56, top: 12, width: 488, height: 320 }) as DOMRect;
    fireEvent.pointerMove(hit, { clientX: 290 });
    expect(document.querySelector(".tooltip")).toHaveTextContent("FPR: 0.5ROC0.8");
    expect(document.querySelector("circle.dot")).toHaveAttribute("cx", "300");
    fireEvent.pointerLeave(hit);
    expect(document.querySelector(".tooltip")).toBeNull();
  });

  it("adds a legend for several series", () => {
    render(
      <LineChart
        title="Loss" description="d" height={200}
        series={[
          { id: "a", label: "Training", tone: "muted", dashed: true, points: [[1, 1], [2, 0.5]] },
          { id: "b", label: "Validation", tone: "accent", points: [[1, 1.1], [2, 0.7]] },
        ]}
        xDomain={[1, 2]} yDomain={[0, 1.2]} xLabel="Epoch" yLabel="Loss" xFormat={String} yFormat={fmt} valueFormat={(v) => v.toFixed(3)}
      />,
    );
    expect(screen.getAllByRole("listitem").map((li) => li.textContent)).toEqual(["Training", "Validation"]);
    expect(document.querySelector("path.line--muted")).toHaveAttribute("stroke-dasharray", "6 4");
  });
});

describe("Histogram", () => {
  it("draws grouped bars per bin with hand-computed geometry", () => {
    render(
      <Histogram
        title="H" description="d" edges={[0, 0.5, 1]} xLabel="p" xFormat={fmt}
        series={[{ label: "Real", tone: "real", mark: "real", counts: [6, 2] }, { label: "AI", tone: "ai", mark: "ai_generated", counts: [1, 8] }]}
        markers={[{ value: 0.5, label: "cut" }]}
      />,
    );
    // plot x 56..544, bins 244 px; slot width (244 - 3*2)/2 = 119; y domain nice(0, 8) = [0, 8], plot y 246..26 (220 px)
    const bars = [...document.querySelectorAll("rect.bar")];
    expect(bars.map((bar) => [bar.getAttribute("x"), bar.getAttribute("width")])).toEqual([
      ["58", "119"], ["302", "119"], ["179", "119"], ["423", "119"],
    ]);
    expect(bars[0]).toHaveAttribute("y", "81"); // 246 - 6/8 * 220
    expect(bars[0]?.querySelector("title")).toHaveTextContent("Real, 0.0–0.5: 6");
    expect(document.querySelector(".marker line")).toHaveAttribute("x1", "300");
    expect(screen.getByText("cut")).toBeInTheDocument();
  });
});

describe("BarList", () => {
  it("scales stacked segments to the largest row", () => {
    render(
      <BarList
        label="Splits"
        rows={[
          { label: "Train", note: "pool", segments: [{ label: "Real", value: 30, tone: "real" }, { label: "AI", value: 10, tone: "ai" }] },
          { label: "Test", segments: [{ label: "Real", value: 20, tone: "accent" }] },
        ]}
      />,
    );
    const rects = [...document.querySelectorAll("rect")];
    expect(rects.map((r) => [r.getAttribute("x"), r.getAttribute("width")])).toEqual([["0%", "75%"], ["75%", "25%"], ["0%", "50%"]]);
    expect([...document.querySelectorAll(".bar-list__value")].map((v) => v.textContent)).toEqual(["30 / 10", "20"]);
    expect(screen.getByText("pool")).toHaveClass("bar-list__note");
    expect(rects[0]?.querySelector("title")).toHaveTextContent("Real: 30");
  });
});

describe("ConfusionMatrix", () => {
  it("labels outcomes and copes with an empty row", () => {
    render(<ConfusionMatrix matrix={[[0, 0], [5, 15]]} caption="20 images" />);
    expect(screen.getByRole("table", { name: "20 images" })).toBeInTheDocument();
    expect(screen.getByText("75.0% of AI-generated images")).toBeInTheDocument();
    expect(screen.getAllByText("0.0% of real images")).toHaveLength(2);
    expect(screen.getByText("True positive").closest("td")).toHaveAttribute("data-correct", "true");
  });
});

describe("Legend", () => {
  it("uses verdict shapes or line samples", () => {
    render(<Legend items={[{ label: "Real", tone: "real", mark: "real" }, { label: "Val", tone: "accent", dashed: true }]} />);
    expect(document.querySelector(".verdict-mark--real")).not.toBeNull();
    expect(document.querySelector("line")).toHaveAttribute("stroke-dasharray", "4 3");
  });
});
