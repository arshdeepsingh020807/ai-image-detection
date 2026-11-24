export type Domain = readonly [number, number];

export interface Scale {
  (value: number): number;
  invert: (pixel: number) => number;
}

export function linearScale([d0, d1]: Domain, [r0, r1]: Domain): Scale {
  const ratio = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const scale = (value: number) => r0 + (value - d0) * ratio;
  scale.invert = (pixel: number) => (ratio === 0 ? d0 : d0 + (pixel - r0) / ratio);
  return scale;
}

const clean = (value: number) => Number(value.toPrecision(12));

export function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const ratio = raw / magnitude;
  const factor = ratio >= Math.sqrt(50) ? 10 : ratio >= Math.sqrt(10) ? 5 : ratio >= Math.sqrt(2) ? 2 : 1;
  return factor * magnitude;
}

export function niceDomain(min: number, max: number, count = 5): Domain {
  if (max <= min) return [min, min + 1];
  const step = niceStep(max - min, count);
  return [clean(Math.floor(min / step) * step), clean(Math.ceil(max / step) * step)];
}

export function ticks([min, max]: Domain, count = 5): number[] {
  if (max <= min) return [min];
  const step = niceStep(max - min, count);
  const first = Math.ceil(clean(min / step));
  const last = Math.floor(clean(max / step));
  return Array.from({ length: last - first + 1 }, (_, i) => clean((first + i) * step));
}

const round = (value: number) => Math.round(value * 100) / 100;

export const linePath = (points: readonly (readonly [number, number])[], x: Scale, y: Scale): string =>
  points.map(([px, py], i) => `${i === 0 ? "M" : "L"}${round(x(px))},${round(y(py))}`).join("");

export function nearestIndex(values: readonly number[], target: number): number {
  let best = 0;
  values.forEach((value, i) => {
    if (Math.abs(value - target) < Math.abs((values[best] ?? Infinity) - target)) best = i;
  });
  return best;
}

export function groupedBar(left: number, right: number, slot: number, slots: number, gap: number): { x: number; width: number } {
  const width = (right - left - gap * (slots + 1)) / slots;
  return { x: left + gap + slot * (width + gap), width: Math.max(0, width) };
}
