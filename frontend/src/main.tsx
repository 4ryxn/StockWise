import { ChangeEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  calculateRecommendation,
  csvTemplate,
  parseLocalSalesCsv,
  populationStandardDeviation,
  seasonalNaiveForecast,
  SafetyMultiplier,
  ServiceLevel,
  UploadedItem,
} from "./localPlanner";
import "./style.css";

type MetricRow = Record<string, string>;
type PlannerMode = "demo" | "own";
type AppView = "planner" | "evidence";
type UploadState = "idle" | "reading" | "success" | "error";

interface Fold { name: string; wape: number }
interface Policy { policy: string; fill_rate: number; average_on_hand_units: number; stockout_units: number }
interface DashboardData {
  overview: { row_count: number; item_count: number; wape: number; improvement: number };
  folds: Fold[]; categories: MetricRow[]; importance: MetricRow[]; inventory: Policy[];
  sensitivity: MetricRow[]; methodology: string[]; note: string;
}
interface PlannerItem { item_id: string; category: string; department: string; forecast: number[]; historical_daily_demand_std: number }
interface PlannerData { forecast_fold: string; forecast_horizon_days: number; items: PlannerItem[] }
interface PlannerInputs { itemId: string; onHand: number; leadTime: number; serviceLevel: ServiceLevel; multiplier: SafetyMultiplier }

const SERVICE_LEVELS: ServiceLevel[] = ["0.9", "0.95", "0.975", "0.99"];
const MULTIPLIERS: SafetyMultiplier[] = ["0.75", "1", "1.25", "1.5", "2"];
const percentage = (value: number | string) => `${(Number(value) * 100).toFixed(2)}%`;
const displayPolicy = (value: string) => value === "stockwise" ? "Forecast-driven" : value.replace("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
const displayFold = (value: string) => value.replace("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
const units = (value: number) => Math.round(value).toLocaleString();
const defaultOnHand = (forecast: number[]) => Math.ceil(forecast.slice(0, 7).reduce((sum, value) => sum + value, 0));

function Bars({ title, rows, keyName, value }: { title: string; rows: MetricRow[] | Fold[]; keyName: string; value: string }) {
  const maximum = Math.max(...rows.map((row) => Number(row[value])));
  return <article className="card chart"><header><h3>{title}</h3><small>BACKTEST WAPE</small></header>{rows.map((row) => {
    const label = String(row[keyName]); const metric = Number(row[value]);
    return <div className="row" key={label} tabIndex={0}><label>{keyName === "name" ? displayFold(label) : displayPolicy(label)}<b>{percentage(metric)}</b></label><i><u style={{ width: `${(metric / maximum) * 100}%` }} title={`${label} ${percentage(metric)}`} /></i></div>;
  })}</article>;
}

function RecommendationCards({
  itemLabel, onHand, leadTimeDemand, safetyStock, recommendedOrder, daysOfCover,
}: {
  itemLabel: string; onHand: number; leadTimeDemand: number; safetyStock: number;
  recommendedOrder: number; daysOfCover: number | null;
}) {
  const atRisk = onHand < leadTimeDemand;
  return <div className="planner-results" aria-live="polite"><article className="recommendation"><small>RECOMMENDED ORDER QUANTITY</small><strong>{units(recommendedOrder)}</strong><span>units for {itemLabel}</span></article><div className="planner-stat-grid"><article><small>EXPECTED LEAD-TIME DEMAND</small><b>{units(leadTimeDemand)} units</b></article><article><small>SAFETY STOCK</small><b>{units(safetyStock)} units</b></article><article><small>CURRENT STOCK / DAYS OF COVER</small><b>{units(onHand)} / {daysOfCover === null ? "—" : `${daysOfCover.toFixed(1)} days`}</b></article><article className={atRisk ? "risk" : "covered"}><small>INVENTORY RISK</small><b>{atRisk ? "Stockout risk" : "Covered for lead time"}</b></article></div></div>;
}

function SettingsControls({ inputs, onChange, itemOptions, onItemChange }: {
  inputs: PlannerInputs; onChange: (next: PlannerInputs) => void;
  itemOptions: { id: string; label: string }[];
  onItemChange: (id: string) => void;
}) {
  return <form className="planner-controls" onSubmit={(event) => event.preventDefault()}>
    <label>Product / SKU<select value={inputs.itemId} onChange={(event) => onItemChange(event.target.value)}>{itemOptions.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
    <label>Current on-hand inventory<input type="number" min="0" step="1" value={inputs.onHand} onChange={(event) => onChange({ ...inputs, onHand: Math.max(0, Number(event.target.value)) })} /></label>
    <label>Lead time<select value={inputs.leadTime} onChange={(event) => onChange({ ...inputs, leadTime: Number(event.target.value) })}>{Array.from({ length: 28 }, (_, index) => index + 1).map((day) => <option key={day} value={day}>{day} day{day === 1 ? "" : "s"}</option>)}</select></label>
    <label>Service level<select value={inputs.serviceLevel} onChange={(event) => onChange({ ...inputs, serviceLevel: event.target.value as ServiceLevel })}>{SERVICE_LEVELS.map((level) => <option key={level} value={level}>{percentage(level)}</option>)}</select></label>
    <label>Safety-stock multiplier<select value={inputs.multiplier} onChange={(event) => onChange({ ...inputs, multiplier: event.target.value as SafetyMultiplier })}>{MULTIPLIERS.map((multiplier) => <option key={multiplier} value={multiplier}>{multiplier}×</option>)}</select></label>
  </form>;
}

function DemoPlanner({ data }: { data: PlannerData }) {
  const first = data.items[0];
  const [inputs, setInputs] = useState<PlannerInputs>({ itemId: first.item_id, onHand: defaultOnHand(first.forecast), leadTime: 7, serviceLevel: "0.95", multiplier: "1" });
  const item = data.items.find((candidate) => candidate.item_id === inputs.itemId) ?? first;
  const result = calculateRecommendation(item.forecast, item.historical_daily_demand_std, inputs.onHand, inputs.leadTime, inputs.serviceLevel, inputs.multiplier);
  return <><div className="planner-grid"><SettingsControls inputs={inputs} onChange={setInputs} itemOptions={data.items.map((candidate) => ({ id: candidate.item_id, label: `${candidate.item_id} · ${candidate.category} / ${candidate.department}` }))} onItemChange={(itemId) => { const next = data.items.find((candidate) => candidate.item_id === itemId) ?? first; setInputs({ ...inputs, itemId: next.item_id, onHand: defaultOnHand(next.forecast) }); }} /><RecommendationCards itemLabel={item.item_id} onHand={inputs.onHand} {...result} /></div><p className="planner-note">This is a scenario recommendation using precomputed held-out LightGBM forecasts for M5 CA_1 items. It is not connected to a live retailer’s inventory system.</p></>;
}

function OwnDataPlanner() {
  const [state, setState] = useState<UploadState>("idle");
  const [items, setItems] = useState<UploadedItem[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [inputs, setInputs] = useState<PlannerInputs>();
  const selected = inputs ? items.find((item) => item.itemId === inputs.itemId) : undefined;
  const forecast = selected ? seasonalNaiveForecast(selected.history) : [];
  const standardDeviation = selected ? populationStandardDeviation(selected.history.map((sale) => sale.unitsSold)) : 0;
  const result = inputs && selected ? calculateRecommendation(forecast, standardDeviation, inputs.onHand, inputs.leadTime, inputs.serviceLevel, inputs.multiplier) : undefined;

  const downloadTemplate = () => {
    const url = URL.createObjectURL(new Blob([csvTemplate()], { type: "text/csv" }));
    const link = document.createElement("a"); link.href = url; link.download = "stockwise-sales-template.csv"; link.click(); URL.revokeObjectURL(url);
  };
  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setState("reading"); setErrors([]); setItems([]); setInputs(undefined);
    const parsed = parseLocalSalesCsv(await file.text());
    if (!parsed.ok) { setErrors(parsed.errors); setState("error"); return; }
    const first = parsed.items[0]; const firstForecast = seasonalNaiveForecast(first.history);
    setItems(parsed.items); setInputs({ itemId: first.itemId, onHand: defaultOnHand(firstForecast), leadTime: 7, serviceLevel: "0.95", multiplier: "1" }); setState("success");
  };
  const selectItem = (itemId: string) => {
    const next = items.find((item) => item.itemId === itemId);
    if (!next || !inputs) return;
    setInputs({ ...inputs, itemId, onHand: defaultOnHand(seasonalNaiveForecast(next.history)) });
  };
  const recordCount = items.reduce((total, item) => total + item.history.length, 0);
  return <div className="own-data"><p className="own-data-intro">Upload daily sales history to create a local inventory scenario. Your file stays in this browser and is never uploaded.</p><div className="upload-line"><label className="upload-zone">Choose CSV<input type="file" accept=".csv,text/csv" onChange={upload} /><span>Required: date, item_id, units_sold</span></label><button type="button" className="template-button" onClick={downloadTemplate}>Download CSV template</button></div><p className="requirements"><b>CSV requirements</b> Required columns: <code>date</code>, <code>item_id</code>, <code>units_sold</code> · Date format: <code>YYYY-MM-DD</code> · At least 28 consecutive daily rows per item · <code>units_sold</code> must be non-negative.</p>{state === "idle" && <p className="planner-empty">Start with the template or choose a CSV. Explicit zero-sales days are required; StockWise never fills missing dates.</p>}{state === "reading" && <p className="upload-status">Checking your file locally…</p>}{state === "error" && <div className="upload-error" role="alert"><b>We could not use that CSV.</b><ul>{errors.map((error) => <li key={error}>{error}</li>)}</ul></div>}{state === "success" && inputs && selected && result && <><p className="upload-status success">Ready: {items.length} valid item{items.length === 1 ? "" : "s"} and {recordCount} daily records processed locally.</p><div className="planner-grid"><SettingsControls inputs={inputs} onChange={setInputs} itemOptions={items.map((item) => ({ id: item.itemId, label: item.itemId }))} onItemChange={selectItem} /><RecommendationCards itemLabel={selected.itemId} onHand={inputs.onHand} {...result} /></div><p className="planner-note">This recommendation uses a local 7-day seasonal-naive forecast from your uploaded daily sales history. It is a planning scenario, not a live demand guarantee.</p><details className="formula"><summary>How this recommendation is calculated</summary><p>The latest 7 daily sales values repeat recursively across a 28-day forecast. Safety stock uses the latest 28 days’ population standard deviation, your service-level z-score, lead time, and safety-stock multiplier.</p></details></>}</div>;
}

function EvidenceSections({ dashboard, selected, serviceLevel, multiplier, setServiceLevel, setMultiplier }: {
  dashboard: DashboardData; selected: MetricRow | undefined; serviceLevel: string; multiplier: string;
  setServiceLevel: (value: string) => void; setMultiplier: (value: string) => void;
}) {
  const categories = dashboard.categories.filter((category) => category.fold === "all_folds");
  return <><section className="evidence"><b>5.9M<small>item-day records</small></b><b>3,049<small>items</small></b><b>3<small>rolling folds</small></b></section><section className="charts"><Bars title="Fold WAPE" rows={dashboard.folds} keyName="name" value="wape" /><Bars title="Category performance" rows={categories} keyName="cat_id" value="wape" /><article className="card chart"><header><h3>Feature importance</h3><small>SPLIT GAIN</small></header>{dashboard.importance.slice(0, 6).map((feature) => <div className="row" key={feature.feature} tabIndex={0}><label>{feature.feature}<b>{Math.round(Number(feature.importance))}</b></label><i><u style={{ width: `${(Number(feature.importance) / 3162) * 100}%` }} /></i></div>)}</article></section><section className="lab"><div className="labhead"><div><p className="eyebrow">PORTFOLIO-LEVEL SCENARIO SIMULATION</p><h2>Inventory policy lab</h2><p>{dashboard.note}</p></div><div className="controls"><label>Service<select value={serviceLevel} onChange={(event) => setServiceLevel(event.target.value)}>{SERVICE_LEVELS.map((level) => <option key={level} value={level}>{percentage(level)}</option>)}</select></label><label>Safety stock<select value={multiplier} onChange={(event) => setMultiplier(event.target.value)}>{MULTIPLIERS.map((value) => <option key={value} value={value}>{value}×</option>)}</select></label></div></div><div className="policies">{dashboard.inventory.map((policy) => <article key={policy.policy}><small>{displayPolicy(policy.policy)}</small><b>{percentage(policy.fill_rate)}</b><span>fill rate</span><hr /><p>{policy.average_on_hand_units.toFixed(1)} avg on-hand · {units(policy.stockout_units)} lost units</p></article>)}</div>{selected && <div className="selected"><small>Selected forecast-driven scenario</small><b>{percentage(selected.fill_rate)}</b><span>{Number(selected.average_on_hand_units).toFixed(1)} on-hand · {units(Number(selected.stockout_units))} lost units</span><p>Trade-off: lower inventory can reduce service level.</p></div>}</section></>;
}

function Methodology({ dashboard }: { dashboard: DashboardData }) {
  return <section className="method"><p className="eyebrow">EVIDENCE &amp; LIMITATIONS</p><h2>How it works</h2>{dashboard.methodology.map((line) => <p key={line}>→ {line}</p>)}</section>;
}

function InventoryPlannerView({ data }: { data: PlannerData }) {
  const [mode, setMode] = useState<PlannerMode>("own");
  return <><section className="product-hero planner-product-hero"><p className="eyebrow">INVENTORY PLANNER</p><h1>Plan your next inventory order.</h1><p>Upload daily sales history, choose an item, and explore a local order recommendation. Your data stays in your browser.</p><span className="privacy-badge">Private · browser-only processing</span><ol className="workflow"><li><b>1</b><span>Upload sales CSV</span></li><li><b>2</b><span>Set inventory assumptions and review recommendation</span></li></ol></section><section className="planner" aria-labelledby="planner-title"><div className="planner-intro"><p className="eyebrow">LOCAL PLANNING SCENARIO</p><h2 id="planner-title">{mode === "own" ? "Use your sales history" : "Explore the M5 model demo"}</h2></div><div className="planner-tabs" role="tablist" aria-label="Planner mode"><button type="button" role="tab" aria-selected={mode === "own"} className={mode === "own" ? "active" : ""} onClick={() => setMode("own")}>Use your own sales data</button><button type="button" role="tab" aria-selected={mode === "demo"} className={mode === "demo" ? "active" : ""} onClick={() => setMode("demo")}>Try M5 model demo</button></div>{mode === "demo" ? <DemoPlanner data={data} /> : <OwnDataPlanner />}</section></>;
}

function ModelEvidenceView({ dashboard, planner, selected, serviceLevel, multiplier, setServiceLevel, setMultiplier }: {
  dashboard: DashboardData; planner: PlannerData; selected: MetricRow | undefined; serviceLevel: string; multiplier: string;
  setServiceLevel: (value: string) => void; setMultiplier: (value: string) => void;
}) {
  return <><section className="product-hero evidence-product-hero"><p className="eyebrow">PORTFOLIO CASE STUDY</p><h1>How StockWise was validated</h1><p>This view contains fixed, held-out M5 CA_1 backtest and scenario-simulation evidence. It demonstrates the project methodology and does not use uploaded visitor data.</p></section><section className="hero evidence-hero"><div><p className="eyebrow">MODEL OPERATIONS / RETAIL FORECASTING</p><h2>Make demand evidence operational.</h2><p>28-day backtests for retail demand, translated into cautious inventory scenario exploration.</p><div className="facts"><b>{percentage(dashboard.overview.wape)}<small>LightGBM WAPE</small></b><b>{dashboard.overview.improvement.toFixed(2)}%<small>relative improvement</small></b></div></div><article className="perf"><small>MODEL PERFORMANCE</small><h2>LightGBM <em>{percentage(dashboard.overview.wape)}</em></h2><label>Seasonal naive <b>87.49%</b></label><i><u style={{ width: "78%" }} /></i><label>LightGBM <b>67.99%</b></label><i><u className="mint" style={{ width: "61%" }} /></i><p>Three held-out rolling folds. Lower WAPE is better.</p></article></section><EvidenceSections dashboard={dashboard} selected={selected} serviceLevel={serviceLevel} multiplier={multiplier} setServiceLevel={setServiceLevel} setMultiplier={setMultiplier} /><section className="demo-evidence"><p className="eyebrow">OPTIONAL MODEL DEMO</p><h2>Explore a model-demo SKU</h2><p>Precomputed held-out M5 forecasts only — not a live retailer system.</p><DemoPlanner data={planner} /></section><Methodology dashboard={dashboard} /></>;
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardData | false>(); const [planner, setPlanner] = useState<PlannerData | false>();
  const [serviceLevel, setServiceLevel] = useState("0.95"); const [multiplier, setMultiplier] = useState("1");
  const [view, setView] = useState<AppView>("planner");
  useEffect(() => { Promise.all([fetch("/data/dashboard.json").then((response) => response.ok ? response.json() as Promise<DashboardData> : Promise.reject()), fetch("/data/planner.json").then((response) => response.ok ? response.json() as Promise<PlannerData> : Promise.reject())]).then(([nextDashboard, nextPlanner]) => { setDashboard(nextDashboard); setPlanner(nextPlanner); }).catch(() => { setDashboard(false); setPlanner(false); }); }, []);
  const selected = useMemo(() => dashboard?.sensitivity.find((scenario) => scenario.service_level === serviceLevel && scenario.safety_stock_multiplier === multiplier), [dashboard, serviceLevel, multiplier]);
  if (!dashboard || !planner) return <main className="state">{dashboard === false ? "Data unavailable — run the static exporter." : "Loading static evidence…"}</main>;
  return <main><nav><div className="brand"><i>SW</i><b>StockWise</b></div><div className="app-nav" role="tablist" aria-label="StockWise views"><button type="button" role="tab" aria-selected={view === "planner"} className={view === "planner" ? "active" : ""} onClick={() => setView("planner")}>Inventory Planner</button><button type="button" role="tab" aria-selected={view === "evidence"} className={view === "evidence" ? "active" : ""} onClick={() => setView("evidence")}>Model Evidence</button></div><mark>{view === "evidence" ? "Static backtest" : "Browser-only"}</mark></nav>{view === "planner" ? <InventoryPlannerView data={planner} /> : <ModelEvidenceView dashboard={dashboard} planner={planner} selected={selected} serviceLevel={serviceLevel} multiplier={multiplier} setServiceLevel={setServiceLevel} setMultiplier={setMultiplier} />}
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
