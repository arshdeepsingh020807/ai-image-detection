import { screen } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import { mockFetch, ok } from "../test/fetchMock";
import { fixtures } from "../test/fixtures";
import { renderAt } from "../test/render";

describe("axe (jsdom, structural rules)", () => {
  it.each([
    ["/", "Detect AI-generated images"],
    ["/performance", "Model performance"],
    ["/method", "Method"],
    ["/missing", "Page not found"],
  ])("%s has no violations", async (path, heading) => {
    mockFetch({ "/api/health": ok(fixtures.health()), "/api/model": ok(fixtures.model()), "/api/metrics": ok(fixtures.metricsFull()) });
    renderAt(path);
    await screen.findByRole("heading", { level: 1, name: heading });
    if (path === "/performance") await screen.findByRole("region", { name: "Headline metrics" });
    const results = await axe.run(document.body, { rules: { "color-contrast": { enabled: false }, region: { enabled: false } } });
    expect(results.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(" ")).join(", ")}`)).toEqual([]);
  });
});
