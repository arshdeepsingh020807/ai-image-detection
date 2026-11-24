import { afterEach, describe, expect, it, vi } from "vitest";
import { applyTheme, readStoredTheme, storeTheme, systemTheme } from "./theme";

describe("theme storage", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("round-trips a stored theme and ignores junk", () => {
    expect(readStoredTheme()).toBeNull();
    storeTheme("dark");
    expect(readStoredTheme()).toBe("dark");
    window.localStorage.setItem("theme", "purple");
    expect(readStoredTheme()).toBeNull();
  });

  it("survives blocked storage", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
    expect(() => { storeTheme("dark"); }).not.toThrow();
    expect(readStoredTheme()).toBeNull();
  });

  it("follows the system preference when matchMedia exists", () => {
    expect(systemTheme()).toBe("light");
    vi.stubGlobal("matchMedia", (query: string) => ({ matches: query.includes("dark"), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    expect(systemTheme()).toBe("dark");
  });

  it("sets and clears the data-theme attribute", () => {
    applyTheme("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    applyTheme(null);
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });
});
