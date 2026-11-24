import { describe, expect, it } from "vitest";
import { fixtures } from "../test/fixtures";
import { fileName, METRIC_KEYS, readAllMetrics, readMetric } from "./metrics";

describe("readMetric", () => {
  it("returns ok, missing or invalid per file", () => {
    const response = fixtures.metricsFull();
    expect(readMetric(response, "roc").status).toBe("ok");
    delete response.metrics.roc;
    expect(readMetric(response, "roc")).toEqual({ status: "missing" });
    response.metrics.pr = { precision: "lots" };
    expect(readMetric(response, "pr")).toEqual({ status: "invalid" });
  });

  it("reads every known file and ignores unknown extras", () => {
    const response = fixtures.metricsFull();
    response.metrics.surprise = { anything: 1 };
    const states = readAllMetrics(response);
    expect(Object.keys(states).sort()).toEqual([...METRIC_KEYS].sort());
    expect(Object.values(states).every((state) => state.status === "ok")).toBe(true);
    expect(Object.values(readAllMetrics(fixtures.metricsEmpty())).every((state) => state.status === "missing")).toBe(true);
  });

  it("names the source file", () => {
    expect(fileName("dataset_report")).toBe("dataset_report.json");
  });
});
