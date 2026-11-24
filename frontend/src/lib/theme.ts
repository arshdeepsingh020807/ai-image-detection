export type Theme = "light" | "dark";

const STORAGE_KEY = "theme";
const DARK_QUERY = "(prefers-color-scheme: dark)";

export function readStoredTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch {
    return null;
  }
}

export function storeTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* storage unavailable (private mode): the choice lasts for this page only */
  }
}

export const systemQuery = (): MediaQueryList | null =>
  typeof window.matchMedia === "function" ? window.matchMedia(DARK_QUERY) : null;

export const systemTheme = (): Theme => (systemQuery()?.matches ? "dark" : "light");

export function applyTheme(theme: Theme | null): void {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}
