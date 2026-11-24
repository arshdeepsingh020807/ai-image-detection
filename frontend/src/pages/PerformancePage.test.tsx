import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { MetricKey } from "../lib/metrics";
import { mockFetch, ok, type Route } from "../test/fetchMock";
import { fixtures } from "../test/fixtures";
import { PerformancePage } from "./PerformancePage";

const SECTIONS: [MetricKey, string][] = [
  ["summary", "headline"], ["confusion", "confusion"], ["confidence", "confidence"], ["history", "history"],
  ["external", "external"], ["robustness", "robustness"], ["dataset_report", "dataset"], ["model_comparison", "comparison"], ["latency", "latency"],
];

function renderPage(metrics: Route | (() => Route), model: Route = ok(fixtures.model())) {
  const mocks = mockFetch({ "/api/metrics": metrics, "/api/model": model });
  render(<PerformancePage />);
  return mocks;
}

const section = (id: string) => document.getElementById(id) as HTMLElement;
const without = (...keys: string[]) => {
  const response = fixtures.metricsFull();
  const metrics = Object.fromEntries(Object.entries(response.metrics).filter(([key]) => !keys.includes(key)));
  return { available: Object.keys(metrics), metrics };
};

describe("PerformancePage with every file published", () => {
  it("renders headline metrics with n and split", async () => {
    renderPage(ok(fixtures.metricsFull()));
    const headline = await screen.findByRole("region", { name: "Headline metrics" });
    expect(within(headline).getByText(/Test split · n = 16,133 images · threshold p\(AI\) ≥ 0\.50/)).toBeInTheDocument();
    const tiles = headline.querySelector(".stat-tiles") as HTMLElement;
    for (const [label, value] of [["Accuracy", "95.1%"], ["Precision", "94.6%"], ["Recall", "95.0%"], ["F1 score", "94.8%"], ["ROC-AUC", "0.987"], ["Specificity", "95.1%"], ["False positive rate", "4.9%"]]) {
      expect(within(tiles).getByText(label as string).nextElementSibling).toHaveTextContent(value as string);
    }
    expect(within(headline).getByRole("table", { name: "Test split with the inconclusive band" })).toHaveTextContent("Coverage (decided)91.2%");
    expect(screen.queryByText("No evaluation results published yet")).not.toBeInTheDocument();
  });

  it("renders the confusion matrix as a table with row shares", async () => {
    renderPage(ok(fixtures.metricsFull()));
    const table = await screen.findByRole("table", { name: "16,133 images" });
    expect(within(table).getByText("8,100")).toBeInTheDocument();
    expect(within(table).getByText("4.9% of real images")).toBeInTheDocument();
    expect(within(table).getByText("False negative")).toBeInTheDocument();
  });

  it("renders ROC, PR, histogram and training charts with accessible names", async () => {
    renderPage(ok(fixtures.metricsFull()));
    expect(await screen.findByRole("img", { name: "ROC curve, AUC 0.987" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Precision–recall curve, average precision 0.985" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Histogram of p(AI) by true class" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Training and validation loss per epoch" })).toBeInTheDocument();
    expect(screen.getByText("Epochs 1–2: train the new classification head on a frozen backbone; Epochs 3–10: fine-tune the backbone at a lower learning rate.")).toBeInTheDocument();
    expect(within(section("confidence")).getByText("low 0.20")).toBeInTheDocument();
  });

  it("renders the external, robustness, dataset, comparison, latency and model card sections", async () => {
    renderPage(ok(fixtures.metricsFull()));
    await screen.findByRole("region", { name: "External test sets" });
    expect(within(section("external")).getByText("External sets were excluded from training, validation and threshold selection.")).toBeInTheDocument();
    expect(within(section("robustness")).getByText("JPEG recompression at quality 50")).toBeInTheDocument();
    expect(within(section("robustness")).getByText("—")).toBeInTheDocument();
    expect(within(section("dataset")).getByText("131,927")).toBeInTheDocument();
    expect(within(section("dataset")).getByText(/Leakage check: no duplicate/)).toBeInTheDocument();
    expect(within(section("comparison")).getByText("Deployed").closest("tr")).toHaveAttribute("data-highlight", "true");
    expect(within(section("latency")).getByText("38.2 ms")).toBeInTheDocument();
    expect(await within(section("model-card")).findByText("fixture-v1")).toBeInTheDocument();
    expect(within(section("model-card")).getByText("Trained epochs")).toBeInTheDocument();
  });
});

describe("PerformancePage honesty", () => {
  it("shows a page-level and per-section empty state when nothing is published", async () => {
    renderPage(ok(fixtures.metricsEmpty()));
    expect(await screen.findByRole("heading", { name: "No evaluation results published yet" })).toBeInTheDocument();
    for (const [key, id] of SECTIONS) {
      expect(within(section(id)).getByText("No evaluation results published yet")).toBeInTheDocument();
      expect(within(section(id)).getByText(`${key}.json`)).toBeInTheDocument();
    }
    expect(within(section("curves")).getAllByText("No evaluation results published yet")).toHaveLength(2);
    expect(document.querySelectorAll(".report-section svg[role=img]")).toHaveLength(0);
    expect(screen.queryByText(/%$/)).not.toBeInTheDocument();
  });

  it.each(SECTIONS)("shows only %s.json as missing when it alone is absent", async (key, id) => {
    renderPage(ok(without(key)));
    await screen.findByRole("region", { name: "Headline metrics" });
    expect(within(section(id)).getByText("No evaluation results published yet")).toBeInTheDocument();
    expect(document.querySelectorAll(".empty-state")).toHaveLength(1);
    expect(screen.queryByRole("heading", { name: "No evaluation results published yet" })).not.toBeInTheDocument();
  });

  it.each(SECTIONS)("flags a malformed %s.json instead of rendering it", async (key, id) => {
    const response = fixtures.metricsFull();
    response.metrics[key] = { unexpected: true };
    renderPage(ok(response));
    await screen.findByRole("region", { name: "Headline metrics" });
    expect(within(section(id)).getByText("Results file could not be read")).toBeInTheDocument();
    expect(document.querySelectorAll(".empty-state")).toHaveLength(1);
  });

  it("omits band markers on the histogram when summary.json is missing", async () => {
    renderPage(ok(without("summary")));
    expect(await screen.findByText("Decision band not shown: summary.json is not published.")).toBeInTheDocument();
  });

  it("handles a training log with no epochs", async () => {
    const response = fixtures.metricsFull();
    response.metrics.history = { epochs: [] };
    renderPage(ok(response));
    expect(await screen.findByText("The training log contains no epochs.")).toBeInTheDocument();
  });

  it("offers a retry when the metrics request fails", async () => {
    let attempts = 0;
    renderPage(() => (++attempts === 1 ? "network" : ok(fixtures.metricsFull())));
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByText("Could not reach the server")).toBeInTheDocument();
    await userEvent.click(within(alert).getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("region", { name: "Headline metrics" })).toBeInTheDocument();
    expect(attempts).toBe(2);
  });

  it("keeps the metrics when the model card cannot be loaded", async () => {
    renderPage(ok(fixtures.metricsFull()), { status: 503, body: { error: { code: "model_unavailable", message: "x" } } });
    expect(await within(await screen.findByRole("region", { name: "Model card" })).findByText("The detection model is unavailable")).toBeInTheDocument();
    expect(within(section("comparison")).queryByText("Deployed")).not.toBeInTheDocument();
    expect(within(section("headline")).getAllByText("95.1%", { selector: ".stat-tile__value" })).toHaveLength(2);
  });

  it("shows a loading state first", () => {
    renderPage("hang");
    expect(screen.getByRole("status")).toHaveTextContent("Loading evaluation results…");
  });
});
