import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyTheme, readStoredTheme } from "./lib/theme";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/detect.css";
import "./styles/report.css";

applyTheme(readStoredTheme());

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
