import { describe, expect, it } from "vitest";
import { ApiError } from "./client";
import { friendlyMessage, toFriendlyError } from "./errors";

const CONTRACT_CODES = [
  "missing_file", "empty_file", "invalid_image", "unsupported_format", "animated_image", "image_too_large", "file_too_large",
  "invalid_request", "rate_limited", "model_unavailable", "not_found", "method_not_allowed", "internal_error",
];
const RETRYABLE = new Set(["missing_file", "invalid_request", "rate_limited", "model_unavailable", "internal_error", "network_error", "timeout", "bad_response"]);

describe("toFriendlyError", () => {
  it.each([...CONTRACT_CODES, "network_error", "timeout", "bad_response"])("has a specific message for %s", (code) => {
    const friendly = toFriendlyError(new ApiError(code, "raw server text"));
    expect(friendly.code).toBe(code);
    expect(friendly.title).not.toBe("The request failed");
    expect(friendly.detail).not.toContain("raw server text");
    expect(friendly.retryable).toBe(RETRYABLE.has(code));
  });

  it("falls back to the server's message for an unknown code", () => {
    expect(toFriendlyError(new ApiError("teapot", "I am a teapot"))).toEqual({ code: "teapot", title: "The request failed", detail: "I am a teapot", retryable: true });
  });

  it("wraps unexpected exceptions without pretending they are network errors", () => {
    expect(toFriendlyError(new TypeError("boom"))).toMatchObject({ code: "unexpected", detail: "boom" });
    expect(toFriendlyError("weird")).toMatchObject({ code: "unexpected", detail: "Request failed." });
  });
});

describe("friendlyMessage", () => {
  it("keeps the default detail or replaces it with context", () => {
    expect(friendlyMessage("empty_file").detail).toMatch(/no data/);
    expect(friendlyMessage("empty_file", "custom").detail).toBe("custom");
  });
});
