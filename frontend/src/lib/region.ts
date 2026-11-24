import type { Region } from "../api/types";

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function containBox(frameAspect: number, imageAspect: number): Box {
  if (imageAspect > frameAspect) {
    const height = (frameAspect / imageAspect) * 100;
    return { left: 0, top: (100 - height) / 2, width: 100, height };
  }
  const width = (imageAspect / frameAspect) * 100;
  return { left: (100 - width) / 2, top: 0, width, height: 100 };
}

export const overlayBox = (frame: Box, region: Region): Box => ({
  left: frame.left + region.x * frame.width,
  top: frame.top + region.y * frame.height,
  width: region.width * frame.width,
  height: region.height * frame.height,
});

export const toPercentStyle = (box: Box) => ({
  left: `${box.left}%`,
  top: `${box.top}%`,
  width: `${box.width}%`,
  height: `${box.height}%`,
});
