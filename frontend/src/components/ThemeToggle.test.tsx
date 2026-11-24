import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ThemeToggle } from "./ThemeToggle";

function fakeSystem(dark: boolean) {
  const listeners = new Set<() => void>();
  const query = {
    matches: dark,
    addEventListener: (_: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_: string, listener: () => void) => listeners.delete(listener),
  };
  vi.stubGlobal("matchMedia", () => query);
  return { change: (next: boolean) => { query.matches = next; listeners.forEach((l) => { l(); }); }, listeners };
}

describe("ThemeToggle", () => {
  it("follows the system theme until the user chooses", () => {
    const system = fakeSystem(false);
    render(<ThemeToggle />);
    const toggle = screen.getByRole("button", { name: "Dark theme" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    act(() => { system.change(true); });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(document.documentElement.dataset.theme).toBeUndefined();
  });

  it("persists the choice and applies it to the document", async () => {
    fakeSystem(false);
    const user = userEvent.setup();
    const { unmount } = render(<ThemeToggle />);
    await user.click(screen.getByRole("button", { name: "Dark theme" }));
    expect(window.localStorage.getItem("theme")).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    unmount();
    render(<ThemeToggle />);
    expect(screen.getByRole("button", { name: "Dark theme" })).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Dark theme" }));
    expect(window.localStorage.getItem("theme")).toBe("light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("is keyboard operable", async () => {
    fakeSystem(false);
    const user = userEvent.setup();
    render(<ThemeToggle />);
    await user.tab();
    expect(screen.getByRole("button", { name: "Dark theme" })).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("button", { name: "Dark theme" })).toHaveAttribute("aria-pressed", "true");
  });

  it("unsubscribes from system changes on unmount", () => {
    const system = fakeSystem(false);
    const { unmount } = render(<ThemeToggle />);
    expect(system.listeners.size).toBe(1);
    unmount();
    expect(system.listeners.size).toBe(0);
  });

  it("works without matchMedia", () => {
    render(<ThemeToggle />);
    expect(screen.getByRole("button", { name: "Dark theme" })).toHaveAttribute("aria-pressed", "false");
  });
});
