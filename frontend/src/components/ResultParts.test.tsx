import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import type { DetectState } from "../hooks/useDetector";
import { fixtures } from "../test/fixtures";
import { JPEG_BYTES, makeFile } from "../test/files";
import { LimitationsNote } from "./LimitationsNote";
import { Notice } from "./Notice";
import { ProbabilityBar } from "./ProbabilityBar";
import { ProgressSteps } from "./ProgressSteps";
import { ResultPanel } from "./ResultPanel";
import { TimingTable } from "./TimingTable";
import { UntrainedNotice } from "./UntrainedNotice";
import { VerdictMark } from "./VerdictMark";

const source = { file: makeFile(JPEG_BYTES), url: "blob:x" };
const panel = (state: DetectState) => {
  const handlers = { onRetry: vi.fn(), onReset: vi.fn(), onCancel: vi.fn() };
  render(<ResultPanel state={state} headingRef={createRef()} {...handlers} />);
  return handlers;
};

describe("ProbabilityBar", () => {
  it("splits the bar by calibrated probability and marks the inconclusive band", () => {
    const { container } = render(<ProbabilityBar real={0.25} aiGenerated={0.75} band={{ low: 0.2, high: 0.8 }} />);
    expect((container.querySelector(".probability__real") as HTMLElement).style.width).toBe("25%");
    expect((container.querySelector(".probability__ai") as HTMLElement).style.width).toBe("75%");
    const band = container.querySelector(".probability__band") as HTMLElement;
    // p(AI) runs right to left on this bar: band [0.2, 0.8] spans x = 1 - 0.8 .. 1 - 0.2
    expect(parseFloat(band.style.left)).toBeCloseTo(20);
    expect(parseFloat(band.style.width)).toBeCloseTo(60);
    expect(screen.getByText("Real").closest("div")).toHaveTextContent("25.0%");
  });

  it("draws a single threshold when there is no band", () => {
    const { container } = render(<ProbabilityBar real={0.5} aiGenerated={0.5} band={{ low: 0.3, high: 0.3 }} />);
    expect(container.querySelector(".probability__band")).toBeNull();
    expect((container.querySelector(".probability__threshold") as HTMLElement).style.left).toBe("70%");
  });
});

describe("ResultPanel", () => {
  it("explains what will appear before an image is chosen", () => {
    panel({ phase: "empty", rejection: null });
    expect(screen.getByText(/choose an image to analyse/i)).toBeInTheDocument();
    expect(screen.getByText("Likely real")).toBeInTheDocument();
  });

  it("prompts to analyse once ready", () => {
    panel({ phase: "ready", source });
    expect(screen.getByText(/ready\. select “analyse image”/i)).toBeInTheDocument();
  });

  it("shows progress with cancel while busy", async () => {
    const { onCancel } = panel({ phase: "analysing", source });
    expect(screen.getByText("Analysis in progress").closest("[aria-busy]")).toHaveAttribute("aria-busy", "true");
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancel).toHaveBeenCalled();
  });

  it("offers retry and reset for a retryable failure", async () => {
    const { onRetry, onReset } = panel({ phase: "failed", source, error: { code: "timeout", title: "Slow", detail: "d", retryable: true } });
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    await userEvent.click(screen.getByRole("button", { name: "Choose another image" }));
    expect(onRetry).toHaveBeenCalled();
    expect(onReset).toHaveBeenCalled();
  });

  it("renders the result with a reset action", async () => {
    const { onReset } = panel({ phase: "done", source, result: fixtures.prediction.real() });
    expect(screen.getByRole("heading", { name: "Likely real" })).toBeInTheDocument();
    expect(screen.getByText("1024 × 512 px")).toBeInTheDocument();
    expect(screen.getByText(/^JPEG, 240\.0\sKB$/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Analyse another image" }));
    expect(onReset).toHaveBeenCalled();
  });
});

describe("ProgressSteps", () => {
  it("marks the upload step complete while analysing", () => {
    render(<ProgressSteps phase="analysing" onCancel={vi.fn()} />);
    const steps = screen.getAllByRole("listitem");
    expect(steps.map((step) => step.dataset.state)).toEqual(["complete", "current", "pending"]);
    expect(steps[0]).toHaveTextContent("Upload (complete)");
    expect(screen.getByText("Running the model…")).toBeInTheDocument();
  });
});

describe("TimingTable", () => {
  it("lists each stage and the total", () => {
    render(<TimingTable timing={fixtures.prediction.ai().timing_ms} />);
    const rows = screen.getAllByRole("row").slice(1).map((row) => row.textContent);
    expect(rows).toEqual(["Decode4.2 ms", "Preprocess2.1 ms", "Inference39 ms", "Heatmap11 ms", "Total56 ms"]);
  });
});

describe("notices", () => {
  it("renders a notice with optional focusable heading and actions", () => {
    const ref = createRef<HTMLHeadingElement>();
    render(<Notice tone="danger" role="alert" title="Oops" headingRef={ref} actions={<button type="button">Fix</button>}><p>Body</p></Notice>);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveClass("notice--danger");
    expect(ref.current).toHaveAttribute("tabindex", "-1");
    expect(within(alert).getByRole("button", { name: "Fix" })).toBeInTheDocument();
  });

  it("renders a plain note without a focus target at the requested level", () => {
    render(<Notice tone="info" title="FYI" level={2} />);
    expect(screen.getByRole("note")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "FYI" })).not.toHaveAttribute("tabindex");
  });

  it("names the untrained model version", () => {
    render(<UntrainedNotice version="untrained-standin" />);
    expect(screen.getByRole("note")).toHaveTextContent(/untrained-standin has not been trained/);
  });

  it("states the limitations plainly", () => {
    render(<LimitationsNote />);
    expect(screen.getByRole("complementary", { name: "How to read a result" })).toHaveTextContent(/statistical estimate, not proof/i);
    expect(screen.getByText(/model attention, not evidence/i)).toBeInTheDocument();
  });
});

describe("VerdictMark", () => {
  it.each(["real", "ai_generated", "inconclusive"] as const)("draws a distinct decorative shape for %s", (label) => {
    const { container } = render(<VerdictMark label={label} />);
    const svg = container.querySelector("svg") as SVGElement;
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg).toHaveClass(`verdict-mark--${label}`);
  });

  it("uses a different path per label", () => {
    const paths = (["real", "ai_generated", "inconclusive"] as const).map((label) => render(<VerdictMark label={label} />).container.querySelector("path")?.getAttribute("d"));
    expect(new Set(paths).size).toBe(3);
  });
});
