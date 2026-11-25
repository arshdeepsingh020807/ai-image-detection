const nbsp = " ";

export const formatPercent = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;

/** A model probability for display: never "100.0%" or "0.0%", which would read as certainty. */
export const formatProbability = (value: number): string =>
  value >= 0.9995 ? ">99.9%" : value <= 0.0005 ? "<0.1%" : formatPercent(value);

export const formatDecimal = (value: number | null | undefined, digits = 3): string =>
  value === null || value === undefined ? "—" : value.toFixed(digits);

export const formatCount = (value: number): string => value.toLocaleString("en-GB");

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}${nbsp}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}${nbsp}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}${nbsp}MB`;
}

export const formatMs = (ms: number): string => (ms < 10 ? `${ms.toFixed(1)}${nbsp}ms` : `${Math.round(ms).toLocaleString("en-GB")}${nbsp}ms`);

export const humanize = (key: string): string => {
  const words = key.replace(/_+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};

export function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString("en-GB", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC", timeZoneName: "short" });
}
