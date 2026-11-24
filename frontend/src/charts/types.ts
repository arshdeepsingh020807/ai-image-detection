export type Tone = "accent" | "real" | "ai" | "muted";

export interface Series {
  id: string;
  label: string;
  tone: Tone;
  points: readonly (readonly [number, number])[];
  dashed?: boolean;
}
