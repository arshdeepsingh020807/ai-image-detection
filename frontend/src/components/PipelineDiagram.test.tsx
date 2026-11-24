import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { connector, PipelineDiagram, placeStages, STAGES } from "./PipelineDiagram";

describe("pipeline layout", () => {
  it("snakes four columns so row two runs right to left", () => {
    // box 176 wide + 32 gap = 208 per column; 56 high + 40 gap = 96 per row
    expect(placeStages(4, 176).map(({ x, y }) => [x, y])).toEqual([
      [0, 0], [208, 0], [416, 0], [624, 0],
      [624, 96], [416, 96], [208, 96], [0, 96],
    ]);
  });

  it("stacks a single column", () => {
    expect(placeStages(1, 300).map(({ x, y }) => [x, y])).toEqual(STAGES.map((_, i) => [0, i * 96]));
  });

  it("connects boxes horizontally in either direction and vertically between rows", () => {
    expect(connector({ x: 0, y: 0 }, { x: 208, y: 0 }, 176)).toBe("M176,28H204");
    expect(connector({ x: 624, y: 96 }, { x: 416, y: 96 }, 176)).toBe("M624,124H596");
    expect(connector({ x: 624, y: 0 }, { x: 624, y: 96 }, 176)).toBe("M712,56V92");
  });

  it("renders accessible diagrams naming every step", () => {
    render(<PipelineDiagram />);
    const diagrams = screen.getAllByRole("img");
    expect(diagrams).toHaveLength(2);
    expect(diagrams[0]).toHaveAccessibleName(/^Pipeline: Collect, then Audit and clean, .* then Serve\.$/);
    expect(document.querySelectorAll(".pipeline--wide .pipeline__edge")).toHaveLength(STAGES.length - 1);
  });
});
