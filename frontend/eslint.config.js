import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "coverage", "node_modules"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.strictTypeChecked],
    languageOptions: { parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname } },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "@typescript-eslint/restrict-template-expressions": ["error", { allowNumber: true }],
    },
  },
  {
    files: ["src/api/guards.test.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-dynamic-delete": "off",
    },
  },
  {
    files: ["**/*.{js,mjs}"],
    extends: [js.configs.recommended],
    languageOptions: { globals: { console: "readonly", fetch: "readonly", process: "readonly", URL: "readonly", setTimeout: "readonly", clearTimeout: "readonly", Buffer: "readonly", window: "readonly", document: "readonly", getComputedStyle: "readonly", DataTransfer: "readonly", File: "readonly", Event: "readonly", DragEvent: "readonly", FormData: "readonly", PerformanceObserver: "readonly", atob: "readonly", axe: "readonly" } },
  },
);
