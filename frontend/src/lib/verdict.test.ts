import { describe, expect, it } from "vitest";
import { fixtures } from "../test/fixtures";
import { describeBand, hasBand, LABEL_TEXT, summarise } from "./verdict";

describe("verdict mapping", () => {
  it("labels every class in words", () => {
    expect(LABEL_TEXT).toEqual({ real: "Likely real", ai_generated: "Likely AI-generated", inconclusive: "Inconclusive" });
  });

  it("explains an AI verdict against the upper cut-off", () => {
    expect(summarise(fixtures.prediction.ai())).toBe("The model estimates a 93.0% probability that this image is AI-generated, at or above the 80.0% needed for this verdict.");
  });

  it("explains a real verdict against the lower cut-off", () => {
    expect(summarise(fixtures.prediction.real())).toBe("The model estimates a 12.0% probability that this image is AI-generated, at or below the 20.0% allowed for a “real” verdict.");
  });

  it("explains an inconclusive result with the band", () => {
    expect(summarise(fixtures.prediction.inconclusive())).toMatch(/^The estimated probability of AI generation \(54\.0%\) falls inside the uncertainty band of 20\.0% to 80\.0%\./);
  });

  it("describes a band and a single threshold differently", () => {
    expect(hasBand({ low: 0.2, high: 0.8 })).toBe(true);
    expect(hasBand({ low: 0.5, high: 0.5 })).toBe(false);
    expect(describeBand({ low: 0.2, high: 0.8 })).toContain("≤ 20.0% for “real” or ≥ 80.0%");
    expect(describeBand({ low: 0.5, high: 0.5 })).toBe("Single decision threshold at p(AI) = 50.0%. This model has no inconclusive band.");
  });
});
