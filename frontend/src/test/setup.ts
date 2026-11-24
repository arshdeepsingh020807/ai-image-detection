import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";

const hasDom = typeof window !== "undefined";

beforeEach(() => {
  if (!hasDom) return;
  let next = 0;
  URL.createObjectURL = vi.fn(() => `blob:preview-${++next}`);
  URL.revokeObjectURL = vi.fn();
  window.scrollTo = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
  if (!hasDom) return;
  cleanup();
  window.localStorage.clear();
  delete document.documentElement.dataset.theme;
});
