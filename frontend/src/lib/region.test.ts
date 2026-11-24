import { describe, expect, it } from "vitest";
import { containBox, overlayBox, toPercentStyle, type Box } from "./region";

const STAGE = 4 / 3;
const close = (box: Box, expected: Box) => {
  (Object.keys(expected) as (keyof Box)[]).forEach((key) => { expect(box[key]).toBeCloseTo(expected[key], 6); });
};

describe("containBox", () => {
  it("letterboxes a 2:1 image in a 4:3 stage", () => {
    // height = (4/3) / 2 = 66.667%, centred: top = (100 - 66.667) / 2 = 16.667%
    close(containBox(STAGE, 1024 / 512), { left: 0, top: 16.666667, width: 100, height: 66.666667 });
  });

  it("pillarboxes a 3:4 portrait in a 4:3 stage", () => {
    // width = 0.75 / (4/3) = 56.25%, left = (100 - 56.25) / 2 = 21.875%
    close(containBox(STAGE, 480 / 640), { left: 21.875, top: 0, width: 56.25, height: 100 });
  });

  it("fills the stage when the aspect ratios match", () => {
    close(containBox(STAGE, 800 / 600), { left: 0, top: 0, width: 100, height: 100 });
  });
});

describe("overlayBox", () => {
  it("places the centre crop of a 1024x512 image", () => {
    // Backend: short side 512 -> 256 gives 512x256; 224 crop at (144, 16).
    // region = {x: 144/512, y: 16/256, w: 224/512, h: 224/256} = {0.28125, 0.0625, 0.4375, 0.875}
    const frame = containBox(STAGE, 2);
    const box = overlayBox(frame, { x: 0.28125, y: 0.0625, width: 0.4375, height: 0.875 });
    // left = 0 + 0.28125*100; top = 16.667 + 0.0625*66.667 = 20.833; width = 43.75; height = 0.875*66.667 = 58.333
    close(box, { left: 28.125, top: 20.833333, width: 43.75, height: 58.333333 });
  });

  it("places the centre crop of a 480x640 portrait", () => {
    // 480x640 -> 256x341; crop 224 at ((256-224)/2, (341-224)/2) = (16, 58)
    const region = { x: 16 / 256, y: 58 / 341, width: 224 / 256, height: 224 / 341 };
    const box = overlayBox(containBox(STAGE, 0.75), region);
    // left = 21.875 + 0.0625*56.25 = 25.390625; width = 0.875*56.25 = 49.21875
    close(box, { left: 25.390625, top: (58 / 341) * 100, width: 49.21875, height: (224 / 341) * 100 });
  });

  it("covers the whole frame for a full-image region", () => {
    const frame = containBox(STAGE, 2);
    expect(overlayBox(frame, { x: 0, y: 0, width: 1, height: 1 })).toEqual(frame);
  });
});

describe("toPercentStyle", () => {
  it("converts a box to CSS percentages", () => {
    expect(toPercentStyle({ left: 1, top: 2, width: 3, height: 4 })).toEqual({ left: "1%", top: "2%", width: "3%", height: "4%" });
  });
});
