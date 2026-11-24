import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { withRouter } from "../test/render";
import { MethodPage } from "./MethodPage";

describe("MethodPage", () => {
  it("explains each pipeline stage with a section and contents link", () => {
    withRouter(<MethodPage />);
    const contents = screen.getByRole("navigation", { name: "Sections" });
    for (const name of ["Data", "Cleaning and leakage prevention", "Transfer learning", "Calibration", "The inconclusive band", "Grad-CAM heatmaps", "Serving", "Limitations"]) {
      const heading = screen.getByRole("heading", { level: 2, name });
      expect(within(contents).getByRole("link", { name })).toHaveAttribute("href", `#${heading.closest("section")?.id ?? ""}`);
    }
    expect(screen.getByRole("link", { name: "Performance" })).toHaveAttribute("href", "/performance");
    expect(screen.getByText(/shows where the model looked, not what is fake/i)).toBeInTheDocument();
  });

  it("contains no numeric results", () => {
    withRouter(<MethodPage />);
    expect(document.body.textContent).not.toMatch(/\d+(\.\d+)?%/);
  });
});
