import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { mockFetch, ok, type Route } from "../test/fetchMock";
import { ServiceStatus } from "./ServiceStatus";

const show = (route: Route) => {
  mockFetch({ "/api/health": route });
  render(<ServiceStatus />);
};

describe("ServiceStatus", () => {
  it("shows checking, then the model version", async () => {
    show(ok({ status: "ok", model_loaded: true, model_version: "v7" }));
    expect(screen.getByText("Checking service…")).toBeInTheDocument();
    expect(await screen.findByText("Service online · model v7")).toHaveAttribute("data-tone", "up");
  });

  it("falls back when the version is not reported", async () => {
    show(ok({ status: "ok", model_loaded: true }));
    expect(await screen.findByText("Service online · model loaded")).toBeInTheDocument();
  });

  it("reports a degraded service", async () => {
    show({ status: 503, body: { status: "degraded", model_loaded: false, model_version: null, uptime_seconds: 1 } });
    expect(await screen.findByText("Service degraded · model not loaded")).toHaveAttribute("data-tone", "down");
  });

  it("reports an unreachable service", async () => {
    show("network");
    expect(await screen.findByText("Service unreachable")).toHaveAttribute("data-tone", "down");
  });
});
