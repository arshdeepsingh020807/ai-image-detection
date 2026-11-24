import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self'",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'",
].join("; ");

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  preview: { headers: { "Content-Security-Policy": CONTENT_SECURITY_POLICY } },
  build: { target: "es2022", assetsInlineLimit: 0 },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}", "tests/unit/**/*.test.mjs"],
    restoreMocks: true,
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/*.test.{ts,tsx}", "src/**/__fixtures__/**", "src/test/**", "src/main.tsx", "src/vite-env.d.ts"],
      thresholds: { lines: 85, branches: 85, functions: 85, statements: 85 },
    },
  },
});
