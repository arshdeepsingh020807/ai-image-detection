import { vi } from "vitest";

export type Route = { status: number; body: unknown } | "network" | "hang";

export function mockFetch(routes: Record<string, Route | (() => Route)>) {
  const calls: string[] = [];
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : input.toString();
    calls.push(url);
    const entry = routes[url];
    const route = typeof entry === "function" ? entry() : entry;
    if (!route) return Promise.resolve(new Response(JSON.stringify({ error: { code: "not_found", message: "nope" } }), { status: 404 }));
    if (route === "network") return Promise.reject(new TypeError("Failed to fetch"));
    if (route === "hang") {
      return new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => { reject(new DOMException("Aborted", "AbortError")); });
      });
    }
    const text = typeof route.body === "string" ? route.body : JSON.stringify(route.body);
    return Promise.resolve(new Response(text, { status: route.status }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, calls };
}

export const ok = (body: unknown): Route => ({ status: 200, body });
