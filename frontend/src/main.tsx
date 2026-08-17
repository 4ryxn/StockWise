import { ChangeEvent, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type MetricRow = Record<string, string>;
interface Fold { name: string; wape: number }
interface Policy { policy: string; fill_rate: number; average_on_hand_units: number; stockout_units: number }
interface DashboardData {
  overview: { row_count: number; item_count: number; wape: number; improvement: number };
  folds: Fold[]; categories: MetricRow[]; importance: MetricRow[]; inventory: Policy[];
  sensitivity: MetricRow[]; methodology: string[]; note: string;
}
interface PlannerItem { item_id: string; category: string; department: string; forecast: number[]; historical_daily_demand_std: number }
interface PlannerData { forecast_fold: string; forecast_horizon_days: number; items: PlannerItem[] }
interface PlannerInputs { itemId: string; onHand: number; leadTime: number; serviceLevel: "0.9" | "0.95" | "0.975" | "0.99"; multiplier: "0.75" | "1" | "1.25" | "1.5" | "2" }

const SERVICE_LEVELS: PlannerInputs["serviceLevel"][] = ["0.9", "0.95", "0.975", "0.99"];
const MULTIPLIERS: PlannerInputs["multiplier"][] = ["0.75", "1", "1.25", "1.5", "2"];
const Z_SCORES: Record<PlannerInputs["serviceLevel"], number> = { "0.9": 1.282, "0.95": 1.645, "0.975": 1.96, "0.99": 2.326 };
const percentage = (value: number | string) => `${(Number(value) * 100).toFixed(2)}%`;
const displayPolicy = (value: string) => value === "stockwise" ? "Forecast-driven" : value.replace("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
const displayFold = (value: string) => value.replace("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
const units = (value: number) => Math.round(value).toLocaleString();

function Bars({ title, rows, keyName, value }: { title: string; rows: MetricRow[] | Fold[]; keyName: string; value: string }) {
  const maximum = Math.max(...rows.map((row) => Number(row[value])));
  return <article className="card chart"><header><h3>{title}</h3><small>BACKTEST WAPE</small></header>{rows.map((row) => {
    const label = String(row[keyName]); const metric = Number(row[value]);
    return <div className="row" key={label} tabIndex={0}><label>{keyName === "name" ? displayFold(label) : displayPolicy(label)}<b>{percentage(metric)}</b></label><i><u style={{ width: `${(metric / maximum) * 100}%` }} title={`${label} ${percentage(metric)}`} /></i></div>;
  })}</article>;
}

function Planner({ data }: { data: PlannerData }) {
  const firstItem = data.items[0];
  const defaultOnHand = Math.ceil(firstItem.forecast.slice(0, 7).reduce((sum, value) => sum + value, 0));
  const [inputs, setInputs] = useState<PlannerInputs>({ itemId: firstItem.item_id, onHand: defaultOnHand, leadTime: 7, serviceLevel: "0.95", multiplier: "1" });
  const item = data.items.find((candidate) => candidate.item_id === inputs.itemId) ?? firstItem;
  const leadForecast = item.forecast.slice(0, inputs.leadTime).reduce((sum, value) => sum + value, 0);
  const averageForecast = item.forecast.reduce((sum, value) => sum + value, 0) / item.forecast.length;
  const safetyStock = Z_SCORES[inputs.serviceLevel] * item.historical_daily_demand_std * Math.sqrt(inputs.leadTime) * Number(inputs.multiplier);
  const recommendation = Math.max(0, Math.ceil(leadForecast + safetyStock - inputs.onHand));
  const daysOfCover = averageForecast > 0 ? inputs.onHand / averageForecast : null;
  const chooseItem = (event: ChangeEvent<HTMLSelectElement>) => {
    const next = data.items.find((candidate) => candidate.item_id === event.target.value) ?? firstItem;
    setInputs((current) => ({ ...current, itemId: next.item_id, onHand: Math.ceil(next.forecast.slice(0, 7).reduce((sum, value) => sum + value, 0)) }));
  };
  return <section className="planner" aria-labelledby="planner-title">
    <div className="planner-intro"><p className="eyebrow">ITEM-LEVEL SCENARIO</p><h2 id="planner-title">Retail demand &amp; inventory planner</h2><p>Use a held-out 28-day forecast to explore an order recommendation for a selected CA_1 item.</p></div>
    <div className="planner-grid"><form className="planner-controls" onSubmit={(event) => event.preventDefault()}>
      <label>Product / SKU<select value={inputs.itemId} onChange={chooseItem}>{data.items.map((candidate) => <option key={candidate.item_id} value={candidate.item_id}>{candidate.item_id} · {candidate.category} / {candidate.department}</option>)}</select></label>
      <label>Current on-hand inventory<input type="number" min="0" step="1" value={inputs.onHand} onChange={(event) => setInputs((current) => ({ ...current, onHand: Math.max(0, Number(event.target.value)) }))} /></label>
      <label>Lead time<select value={inputs.leadTime} onChange={(event) => setInputs((current) => ({ ...current, leadTime: Number(event.target.value) }))}>{Array.from({ length: data.forecast_horizon_days }, (_, index) => index + 1).map((day) => <option key={day} value={day}>{day} day{day === 1 ? "" : "s"}</option>)}</select></label>
      <label>Service level<select value={inputs.serviceLevel} onChange={(event) => setInputs((current) => ({ ...current, serviceLevel: event.target.value as PlannerInputs["serviceLevel"] }))}>{SERVICE_LEVELS.map((level) => <option key={level} value={level}>{percentage(level)}</option>)}</select></label>
      <label>Safety-stock multiplier<select value={inputs.multiplier} onChange={(event) => setInputs((current) => ({ ...current, multiplier: event.target.value as PlannerInputs["multiplier"] }))}>{MULTIPLIERS.map((multiplier) => <option key={multiplier} value={multiplier}>{multiplier}×</option>)}</select></label>
    </form><div className="planner-results" aria-live="polite"><article className="recommendation"><small>RECOMMENDED ORDER QUANTITY</small><strong>{units(recommendation)}</strong><span>units for {item.item_id}</span></article><div className="planner-stat-grid"><article><small>EXPECTED LEAD-TIME DEMAND</small><b>{units(leadForecast)} units</b></article><article><small>SAFETY STOCK</small><b>{units(safetyStock)} units</b></article><article><small>CURRENT STOCK / DAYS OF COVER</small><b>{units(inputs.onHand)} / {daysOfCover === null ? "—" : `${daysOfCover.toFixed(1)} days`}</b></article><article className={inputs.onHand < leadForecast ? "risk" : "covered"}><small>INVENTORY RISK</small><b>{inputs.onHand < leadForecast ? "Stockout risk" : "Covered for lead time"}</b></article></div></div></div>
    <p className="planner-note">This is a scenario recommendation using precomputed held-out LightGBM forecasts for M5 CA_1 items. It is not connected to a live retailer’s inventory system.</p>
  </section>;
}

function App() {
  const [dashboard, setDashboard] = useState<DashboardData | false>(); const [planner, setPlanner] = useState<PlannerData | false>();
  const [serviceLevel, setServiceLevel] = useState("0.95"); const [multiplier, setMultiplier] = useState("1");
  useEffect(() => { Promise.all([fetch("/data/dashboard.json").then((response) => response.ok ? response.json() as Promise<DashboardData> : Promise.reject()), fetch("/data/planner.json").then((response) => response.ok ? response.json() as Promise<PlannerData> : Promise.reject())]).then(([nextDashboard, nextPlanner]) => { setDashboard(nextDashboard); setPlanner(nextPlanner); }).catch(() => { setDashboard(false); setPlanner(false); }); }, []);
  if (!dashboard || !planner) return <main className="state">{dashboard === false ? "Data unavailable — run the static exporter." : "Loading static evidence…"}</main>;
  const selected = dashboard.sensitivity.find((scenario) => scenario.service_level === serviceLevel && scenario.safety_stock_multiplier === multiplier);
  const categories = dashboard.categories.filter((category) => category.fold === "all_folds");
  return <main><nav><div className="brand"><i>SW</i><b>StockWise</b></div><span>CA_1 retail demand intelligence</span><mark>Static backtest</mark></nav>
    <section className="hero"><div><p className="eyebrow">MODEL OPERATIONS / RETAIL FORECASTING</p><h1>Make demand evidence operational.</h1><p>28-day backtests for retail demand, translated into cautious inventory scenario exploration.</p><div className="facts"><b>{percentage(dashboard.overview.wape)}<small>LightGBM WAPE</small></b><b>{dashboard.overview.improvement.toFixed(2)}%<small>relative improvement</small></b></div></div><article className="perf"><small>MODEL PERFORMANCE</small><h2>LightGBM <em>{percentage(dashboard.overview.wape)}</em></h2><label>Seasonal naive <b>87.49%</b></label><i><u style={{ width: "78%" }} /></i><label>LightGBM <b>67.99%</b></label><i><u className="mint" style={{ width: "61%" }} /></i><p>Three held-out rolling folds. Lower WAPE is better.</p></article></section>
    <Planner data={planner} /><section className="evidence"><b>5.9M<small>item-day records</small></b><b>3,049<small>items</small></b><b>3<small>rolling folds</small></b></section>
    <section className="charts"><Bars title="Fold WAPE" rows={dashboard.folds} keyName="name" value="wape" /><Bars title="Category performance" rows={categories} keyName="cat_id" value="wape" /><article className="card chart"><header><h3>Feature importance</h3><small>SPLIT GAIN</small></header>{dashboard.importance.slice(0, 6).map((feature) => <div className="row" key={feature.feature} tabIndex={0}><label>{feature.feature}<b>{Math.round(Number(feature.importance))}</b></label><i><u style={{ width: `${(Number(feature.importance) / 3162) * 100}%` }} /></i></div>)}</article></section>
    <section className="lab"><div className="labhead"><div><p className="eyebrow">PORTFOLIO-LEVEL SCENARIO SIMULATION</p><h2>Inventory policy lab</h2><p>{dashboard.note}</p></div><div className="controls"><label>Service<select value={serviceLevel} onChange={(event) => setServiceLevel(event.target.value)}>{SERVICE_LEVELS.map((level) => <option key={level} value={level}>{percentage(level)}</option>)}</select></label><label>Safety stock<select value={multiplier} onChange={(event) => setMultiplier(event.target.value)}>{MULTIPLIERS.map((value) => <option key={value} value={value}>{value}×</option>)}</select></label></div></div><div className="policies">{dashboard.inventory.map((policy) => <article key={policy.policy}><small>{displayPolicy(policy.policy)}</small><b>{percentage(policy.fill_rate)}</b><span>fill rate</span><hr /><p>{policy.average_on_hand_units.toFixed(1)} avg on-hand · {units(policy.stockout_units)} lost units</p></article>)}</div>{selected && <div className="selected"><small>Selected forecast-driven scenario</small><b>{percentage(selected.fill_rate)}</b><span>{Number(selected.average_on_hand_units).toFixed(1)} on-hand · {units(Number(selected.stockout_units))} lost units</span><p>Trade-off: lower inventory can reduce service level.</p></div>}</section>
    <section className="method"><p className="eyebrow">EVIDENCE &amp; LIMITATIONS</p><h2>How it works</h2>{dashboard.methodology.map((line) => <p key={line}>→ {line}</p>)}</section>
  </main>;
}
createRoot(document.getElementById("root")!).render(<App />);
