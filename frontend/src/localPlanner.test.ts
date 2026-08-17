import { describe, expect, it } from "vitest";

import {
  calculateRecommendation,
  DailySale,
  parseLocalSalesCsv,
  seasonalNaiveForecast,
} from "./localPlanner";

const dates = Array.from({ length: 28 }, (_, index) =>
  new Date(Date.UTC(2026, 0, index + 1)).toISOString().slice(0, 10),
);

function csv(itemId: string, values: number[], selectedDates: string[] = dates): string {
  return [
    "date,item_id,units_sold",
    ...selectedDates.map((date, index) => `${date},${itemId},${values[index] ?? 1}`),
  ].join("\n");
}

describe("local planner CSV validation", () => {
  it("parses a valid multi-item CSV", () => {
    const source = [
      "date,item_id,units_sold",
      ...dates.flatMap((date, index) => [`${date},SKU_A,${index}`, `${date},SKU_B,0`]),
    ].join("\n");
    const result = parseLocalSalesCsv(source);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.items).toHaveLength(2);
      expect(result.items[0].history).toHaveLength(28);
    }
  });

  it("rejects duplicate item-date rows", () => {
    const result = parseLocalSalesCsv(`${csv("SKU_A", Array(28).fill(1))}\n${dates[0]},SKU_A,2`);
    expect(result.ok).toBe(false);
  });

  it("rejects a missing daily observation", () => {
    const missingOneDay = dates.filter((_, index) => index !== 10);
    const result = parseLocalSalesCsv(csv("SKU_A", Array(28).fill(1), missingOneDay));
    expect(result.ok).toBe(false);
  });
});

describe("local planner calculations", () => {
  it("repeats the last seven values for a 28-day seasonal-naive forecast", () => {
    const history: DailySale[] = dates.map((date, index) => ({ date, unitsSold: index + 1 }));
    expect(seasonalNaiveForecast(history)).toEqual([
      22, 23, 24, 25, 26, 27, 28,
      22, 23, 24, 25, 26, 27, 28,
      22, 23, 24, 25, 26, 27, 28,
      22, 23, 24, 25, 26, 27, 28,
    ]);
  });

  it("calculates a clipped inventory recommendation", () => {
    const result = calculateRecommendation(Array(28).fill(10), 2, 20, 2, "0.95", "1");
    expect(result.leadTimeDemand).toBe(20);
    expect(result.safetyStock).toBeCloseTo(4.6528, 4);
    expect(result.recommendedOrder).toBe(5);
    expect(result.daysOfCover).toBe(2);
  });
});
