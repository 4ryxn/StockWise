export type ServiceLevel = "0.9" | "0.95" | "0.975" | "0.99";
export type SafetyMultiplier = "0.75" | "1" | "1.25" | "1.5" | "2";

export interface DailySale {
  date: string;
  unitsSold: number;
}

export interface UploadedItem {
  itemId: string;
  history: DailySale[];
}

export type CsvParseResult =
  | { ok: true; items: UploadedItem[] }
  | { ok: false; errors: string[] };

export interface Recommendation {
  leadTimeDemand: number;
  safetyStock: number;
  recommendedOrder: number;
  daysOfCover: number | null;
}

const REQUIRED_COLUMNS = ["date", "item_id", "units_sold"] as const;
const Z_SCORES: Record<ServiceLevel, number> = {
  "0.9": 1.282,
  "0.95": 1.645,
  "0.975": 1.96,
  "0.99": 2.326,
};

function parseCsvRows(input: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index];
    if (quoted) {
      if (character === '"' && input[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        cell += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(cell);
      cell = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && input[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }
  if (quoted) throw new Error("The CSV has an unterminated quoted value.");
  row.push(cell);
  if (row.some((value) => value.trim() !== "")) rows.push(row);
  return rows;
}

function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function isNextDay(previous: string, current: string): boolean {
  const previousDate = new Date(`${previous}T00:00:00Z`);
  const currentDate = new Date(`${current}T00:00:00Z`);
  return currentDate.valueOf() - previousDate.valueOf() === 86_400_000;
}

export function parseLocalSalesCsv(input: string): CsvParseResult {
  let rows: string[][];
  try {
    rows = parseCsvRows(input.replace(/^\uFEFF/, ""));
  } catch (error) {
    return { ok: false, errors: [error instanceof Error ? error.message : "Unable to read CSV."] };
  }
  if (rows.length < 2) return { ok: false, errors: ["Provide a header row and at least one sales row."] };

  const header = rows[0].map((value) => value.trim().toLowerCase());
  const indexes = REQUIRED_COLUMNS.map((column) => header.indexOf(column));
  const hasExactlyOneOfEach = REQUIRED_COLUMNS.every(
    (column) => header.filter((value) => value === column).length === 1,
  );
  if (indexes.some((index) => index < 0) || !hasExactlyOneOfEach) {
    return { ok: false, errors: ["Required columns are date, item_id, and units_sold."] };
  }
  const [dateIndex, itemIndex, unitsIndex] = indexes;
  const grouped = new Map<string, DailySale[]>();
  const errors: string[] = [];
  const seen = new Set<string>();
  rows.slice(1).forEach((row, offset) => {
    const line = offset + 2;
    const date = (row[dateIndex] ?? "").trim();
    const itemId = (row[itemIndex] ?? "").trim();
    const unitsValue = (row[unitsIndex] ?? "").trim();
    const unitsSold = Number(unitsValue);
    if (!isIsoDate(date)) errors.push(`Row ${line}: date must use YYYY-MM-DD.`);
    if (!itemId) errors.push(`Row ${line}: item_id is required.`);
    if (!Number.isFinite(unitsSold) || unitsValue === "" || unitsSold < 0) {
      errors.push(`Row ${line}: units_sold must be a non-negative number.`);
    }
    const key = `${itemId}\u0000${date}`;
    if (itemId && date && seen.has(key)) errors.push(`Row ${line}: duplicate item_id and date.`);
    seen.add(key);
    if (errors.length === 0 || (isIsoDate(date) && itemId && Number.isFinite(unitsSold) && unitsSold >= 0)) {
      const sales = grouped.get(itemId) ?? [];
      sales.push({ date, unitsSold });
      grouped.set(itemId, sales);
    }
  });
  if (errors.length > 0) return { ok: false, errors: [...new Set(errors)].slice(0, 8) };

  const validItems: UploadedItem[] = [];
  grouped.forEach((sales, itemId) => {
    const sorted = [...sales].sort((left, right) => left.date.localeCompare(right.date));
    const latest = sorted.slice(-28);
    const consecutive = latest.length === 28 && latest.every((sale, index) => index === 0 || isNextDay(latest[index - 1].date, sale.date));
    if (!consecutive) {
      errors.push(`${itemId}: provide 28 consecutive daily rows through the most recent date; include explicit zero-sales days.`);
    } else {
      validItems.push({ itemId, history: latest });
    }
  });
  if (errors.length > 0) return { ok: false, errors: errors.slice(0, 8) };
  return { ok: true, items: validItems.sort((left, right) => left.itemId.localeCompare(right.itemId)) };
}

export function seasonalNaiveForecast(history: DailySale[]): number[] {
  const values = history.slice(-28).map((sale) => sale.unitsSold);
  if (values.length < 28) throw new Error("At least 28 daily observations are required.");
  const forecast = [...values.slice(-7)];
  while (forecast.length < 28) forecast.push(forecast[forecast.length - 7]);
  return forecast;
}

export function populationStandardDeviation(values: number[]): number {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

export function calculateRecommendation(
  forecast: number[],
  historicalStandardDeviation: number,
  onHand: number,
  leadTime: number,
  serviceLevel: ServiceLevel,
  multiplier: SafetyMultiplier,
): Recommendation {
  const leadTimeDemand = forecast.slice(0, leadTime).reduce((sum, value) => sum + value, 0);
  const safetyStock = Z_SCORES[serviceLevel] * historicalStandardDeviation * Math.sqrt(leadTime) * Number(multiplier);
  const averageForecast = forecast.reduce((sum, value) => sum + value, 0) / forecast.length;
  return {
    leadTimeDemand,
    safetyStock,
    recommendedOrder: Math.max(0, Math.ceil(leadTimeDemand + safetyStock - onHand)),
    daysOfCover: averageForecast > 0 ? onHand / averageForecast : null,
  };
}

export function csvTemplate(): string {
  const start = new Date("2026-01-01T00:00:00Z");
  const rows = Array.from({ length: 28 }, (_, index) => {
    const date = new Date(start.valueOf() + index * 86_400_000).toISOString().slice(0, 10);
    return `${date},SKU_001,${index % 7 === 1 ? 0 : 12}`;
  });
  return `date,item_id,units_sold\n${rows.join("\n")}\n`;
}
