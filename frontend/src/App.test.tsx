import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { mockFetch, ok } from "./test/fetchMock";
import { fixtures } from "./test/fixtures";
import { renderAt } from "./test/render";

describe("routing and layout", () => {
  beforeEach(() => {
    mockFetch({ "/api/health": ok(fixtures.health()), "/api/model": ok(fixtures.model()), "/api/metrics": ok(fixtures.metricsEmpty()) });
  });

  it.each([
    ["/", "Detect AI-generated images", "Detect"],
    ["/performance", "Model performance", "Performance"],
    ["/method", "Method", "Method"],
  ])("renders %s with the matching nav item marked current", async (path, heading, nav) => {
    renderAt(path);
    expect(screen.getByRole("heading", { level: 1, name: heading })).toBeInTheDocument();
    const navigation = screen.getByRole("navigation", { name: "Main" });
    expect(within(navigation).getByRole("link", { name: nav })).toHaveAttribute("aria-current", "page");
    expect(within(navigation).getAllByRole("link").filter((link) => link.hasAttribute("aria-current"))).toHaveLength(1);
    await screen.findByText(/Service online/);
  });

  it("renders a 404 page for unknown paths with links back", () => {
    renderAt("/no/such/page");
    expect(screen.getByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByText("/no/such/page")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Analyse an image" })).toHaveAttribute("href", "/");
    expect(document.title).toBe("Page not found · AI Image Detection");
  });

  it("provides landmarks and a skip link to main content", () => {
    renderAt("/");
    expect(screen.getByRole("link", { name: "Skip to main content" })).toHaveAttribute("href", "#main");
    expect(screen.getByRole("main")).toHaveAttribute("id", "main");
    expect(screen.getByRole("banner")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
  });

  it("moves focus to the main region after client-side navigation", async () => {
    const scrollTo = vi.spyOn(window, "scrollTo");
    const user = userEvent.setup();
    renderAt("/");
    await user.click(screen.getByRole("link", { name: "Method" }));
    expect(screen.getByRole("heading", { level: 1, name: "Method" })).toBeInTheDocument();
    await waitFor(() => { expect(screen.getByRole("main")).toHaveFocus(); });
    expect(scrollTo).toHaveBeenCalledWith(0, 0);
  });
});

describe("App", () => {
  it("mounts with the browser router at the current URL", async () => {
    mockFetch({ "/api/health": ok(fixtures.health()), "/api/model": ok(fixtures.model()) });
    window.history.pushState({}, "", "/method");
    render(<App />);
    expect(screen.getByRole("heading", { level: 1, name: "Method" })).toBeInTheDocument();
    await screen.findByText(/Service online/);
    window.history.pushState({}, "", "/");
  });
});
