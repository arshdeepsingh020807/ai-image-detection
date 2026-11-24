import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FakeXHR, installFakeXhr } from "../test/fakeXhr";
import { mockFetch, ok } from "../test/fetchMock";
import { fixtures } from "../test/fixtures";
import { makeFile, JPEG_BYTES } from "../test/files";
import { ApiError, getHealth, getMetrics, getModel, parseResponse, predict, PREDICT_TIMEOUT_MS, READ_TIMEOUT_MS } from "./client";
import { isHealth } from "./guards";

const expectApiError = async (promise: Promise<unknown>, code: string, status?: number) => {
  const error = await promise.then(() => null, (e: unknown) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect((error as ApiError).code).toBe(code);
  if (status !== undefined) expect((error as ApiError).status).toBe(status);
};

describe("parseResponse", () => {
  it("returns a body that passes the guard", () => {
    expect(parseResponse(200, JSON.stringify(fixtures.health()), isHealth)).toEqual(fixtures.health());
  });

  it("maps the contract error body to its code and status", () => {
    const run = () => parseResponse(429, JSON.stringify({ error: { code: "rate_limited", message: "Too many" } }), isHealth);
    expect(run).toThrow(expect.objectContaining({ code: "rate_limited", status: 429, message: "Too many" }));
  });

  it("flags a 2xx body that breaks the contract", () => {
    expect(() => parseResponse(200, JSON.stringify({ status: "ok" }), isHealth)).toThrow(expect.objectContaining({ code: "bad_response" }));
  });

  it("flags a non-JSON error page such as a proxy 502", () => {
    expect(() => parseResponse(502, "<html>Bad Gateway</html>", isHealth)).toThrow(expect.objectContaining({ code: "bad_response", status: 502 }));
  });
});

describe("GET helpers", () => {
  afterEach(() => { vi.useRealTimers(); });

  it("fetch the documented paths", async () => {
    const { calls } = mockFetch({ "/api/model": ok(fixtures.model()), "/api/metrics": ok(fixtures.metricsEmpty()) });
    await expect(getModel()).resolves.toMatchObject({ version: "fixture-v1" });
    await expect(getMetrics()).resolves.toEqual({ available: [], metrics: {} });
    expect(calls).toEqual(["/api/model", "/api/metrics"]);
  });

  it("treat a degraded 503 health body as data", async () => {
    const degraded = { status: "degraded", model_loaded: false, model_version: null, uptime_seconds: 4.5 };
    mockFetch({ "/api/health": { status: 503, body: degraded } });
    await expect(getHealth()).resolves.toEqual(degraded);
  });

  it("reject a 503 from other endpoints with its code", async () => {
    mockFetch({ "/api/model": { status: 503, body: { error: { code: "model_unavailable", message: "x" } } } });
    await expectApiError(getModel(), "model_unavailable", 503);
  });

  it("map a failed connection to network_error", async () => {
    mockFetch({ "/api/metrics": "network" });
    await expectApiError(getMetrics(), "network_error");
  });

  it("time out with an AbortController", async () => {
    vi.useFakeTimers();
    mockFetch({ "/api/metrics": "hang" });
    const pending = getMetrics();
    vi.advanceTimersByTime(READ_TIMEOUT_MS);
    await expectApiError(pending, "timeout");
  });

  it("honour a caller's abort signal", async () => {
    mockFetch({ "/api/metrics": "hang" });
    const controller = new AbortController();
    const pending = getMetrics({ signal: controller.signal });
    controller.abort();
    await expectApiError(pending, "aborted");
  });
});

describe("predict", () => {
  const file = makeFile(JPEG_BYTES, "photo.jpg");
  beforeEach(installFakeXhr);

  it("posts multipart form data with the explain flag and reports upload completion", async () => {
    const onUploaded = vi.fn();
    const pending = predict(file, { explain: true, onUploaded });
    const xhr = FakeXHR.latest();
    expect([xhr.method, xhr.url, xhr.timeout]).toEqual(["POST", "/api/predict", PREDICT_TIMEOUT_MS]);
    expect(xhr.body?.get("file")).toBe(file);
    expect(xhr.body?.get("explain")).toBe("true");
    xhr.finishUpload();
    expect(onUploaded).toHaveBeenCalledOnce();
    xhr.respond(200, fixtures.prediction.ai());
    await expect(pending).resolves.toMatchObject({ label: "ai_generated" });
  });

  it("sends explain=false when the heatmap is off", () => {
    void predict(file, { explain: false });
    expect(FakeXHR.latest().body?.get("explain")).toBe("false");
  });

  it.each([
    [400, "invalid_image"], [400, "invalid_request"], [405, "method_not_allowed"], [413, "file_too_large"], [415, "animated_image"], [429, "rate_limited"], [503, "model_unavailable"], [500, "internal_error"],
  ])("maps HTTP %i to %s", async (status, code) => {
    const pending = predict(file, { explain: true });
    FakeXHR.latest().respond(status, { error: { code, message: "server says no" } });
    await expectApiError(pending, code, status);
  });

  it("rejects a 200 that breaks the contract", async () => {
    const pending = predict(file, { explain: true });
    FakeXHR.latest().respond(200, { label: "real" });
    await expectApiError(pending, "bad_response");
  });

  it.each([
    ["network_error", (xhr: FakeXHR) => xhr.onerror?.()],
    ["timeout", (xhr: FakeXHR) => xhr.ontimeout?.()],
  ])("maps transport failure to %s", async (code, fail) => {
    const pending = predict(file, { explain: true });
    fail(FakeXHR.latest());
    await expectApiError(pending, code);
  });

  it("aborts the request when the signal fires", async () => {
    const controller = new AbortController();
    const pending = predict(file, { explain: true, signal: controller.signal });
    controller.abort();
    expect(FakeXHR.latest().aborted).toBe(true);
    await expectApiError(pending, "aborted");
  });

  it("does not send when the signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    await expectApiError(predict(file, { explain: true, signal: controller.signal }), "aborted");
    expect(FakeXHR.instances).toHaveLength(0);
  });
});
