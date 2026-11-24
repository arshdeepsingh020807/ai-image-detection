import { useCallback, useEffect, useState } from "react";
import { applyTheme, readStoredTheme, storeTheme, systemQuery, systemTheme, type Theme } from "../lib/theme";

export function useTheme(): { theme: Theme; toggle: () => void } {
  const [stored, setStored] = useState<Theme | null>(readStoredTheme);
  const [system, setSystem] = useState<Theme>(systemTheme);

  useEffect(() => {
    const query = systemQuery();
    if (!query) return;
    const update = () => { setSystem(query.matches ? "dark" : "light"); };
    query.addEventListener("change", update);
    return () => { query.removeEventListener("change", update); };
  }, []);

  useEffect(() => { applyTheme(stored); }, [stored]);

  const theme = stored ?? system;
  const toggle = useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    storeTheme(next);
    setStored(next);
  }, [theme]);

  return { theme, toggle };
}
