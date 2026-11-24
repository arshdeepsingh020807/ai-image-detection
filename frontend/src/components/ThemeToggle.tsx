import { useTheme } from "../hooks/useTheme";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button type="button" className="theme-toggle" aria-pressed={theme === "dark"} onClick={toggle}>
      <span className="theme-toggle__track" aria-hidden="true"><span className="theme-toggle__thumb" /></span>
      Dark theme
    </button>
  );
}
