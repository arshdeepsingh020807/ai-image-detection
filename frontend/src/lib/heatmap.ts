export const HEATMAP_MAX_ALPHA = 200;

export const HEATMAP_STOPS = [
  { at: 0, rgb: [120, 28, 109] },
  { at: 1 / 3, rgb: [212, 72, 66] },
  { at: 2 / 3, rgb: [250, 153, 8] },
  { at: 1, rgb: [252, 255, 164] },
] as const;

const round = (value: number, digits: number) => Number(value.toFixed(digits));

export const stopAlpha = (at: number): number => round((at * HEATMAP_MAX_ALPHA) / 255, 3);

export const heatmapGradient = (): string =>
  `linear-gradient(90deg, ${HEATMAP_STOPS.map(({ at, rgb }) => `rgba(${rgb.join(", ")}, ${stopAlpha(at)}) ${round(at * 100, 2)}%`).join(", ")})`;
