import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { fixtures } from "../test/fixtures";
import { loadImage } from "../test/render";
import { ImageStage } from "./ImageStage";

const explanation = fixtures.prediction.ai().explanation;
if (!explanation) throw new Error("fixture needs an explanation");

const renderStage = (overrides: Partial<Parameters<typeof ImageStage>[0]> = {}) => {
  const props = { src: "blob:x", alt: "Preview", explanation, showHeatmap: true, opacity: 0.6, onDimensions: vi.fn(), onDecodeError: vi.fn(), ...overrides };
  render(<ImageStage {...props} />);
  return props;
};

describe("ImageStage", () => {
  it("keeps the image hidden and unpositioned until it has loaded", () => {
    renderStage();
    expect(screen.getByAltText("Preview").getAttribute("style")).toBeNull();
    expect(screen.queryByTestId("heatmap-region")).not.toBeInTheDocument();
  });

  it("pillarboxes a portrait image and reports its natural size", () => {
    const props = renderStage({ explanation: null });
    const img = screen.getByAltText("Preview");
    loadImage(img, 480, 640);
    expect(props.onDimensions).toHaveBeenCalledWith(480, 640);
    expect(img.style.left).toBe("21.875%");
    expect(img.style.width).toBe("56.25%");
    expect(img.style.height).toBe("100%");
  });

  it("positions the region relative to the displayed image", () => {
    renderStage({ explanation: { ...explanation, region: { x: 0.25, y: 0.5, width: 0.5, height: 0.5 } } });
    loadImage(screen.getByAltText("Preview"), 480, 640);
    const region = screen.getByTestId("heatmap-region");
    // frame left 21.875%, width 56.25%: left = 21.875 + 0.25*56.25 = 35.9375; width = 28.125; top = 50; height = 50
    expect(region.style.left).toBe("35.9375%");
    expect(region.style.width).toBe("28.125%");
    expect(region.style.top).toBe("50%");
    expect(region.style.height).toBe("50%");
  });

  it("keeps the region outline but hides the heatmap when asked", () => {
    renderStage({ showHeatmap: false });
    loadImage(screen.getByAltText("Preview"), 1024, 512);
    expect(screen.getByTestId("heatmap-region")).toBeEmptyDOMElement();
  });

  it("reports decode errors", () => {
    const props = renderStage();
    screen.getByAltText("Preview").dispatchEvent(new Event("error"));
    expect(props.onDecodeError).toHaveBeenCalled();
  });
});
