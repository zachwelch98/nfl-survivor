import { Bar, BarChart, CartesianGrid, Legend, Line, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { AsyncView } from "../components/AsyncView";
import { Card, StatTile } from "../components/Card";
import { useAsync } from "../lib/useAsync";
import { getBacktest, getCalibration } from "../data/source";
import type { BacktestMethodSummary, Calibration, ConfidenceSlice } from "../data/types";

const VEGAS_COLOR = "#2563eb";
const MC_COLOR = "#f59e0b";
const ML_COLOR = "#16a34a";

export default function AnalysisView() {
  const calibration = useAsync(() => getCalibration(), []);
  const backtest = useAsync(() => getBacktest(), []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Historical Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">Which method is actually better calibrated, backed by evidence rather than assumed.</p>
      </div>

      <AsyncView state={calibration}>
        {(c) => (
          <>
            <Card title="Overall calibration" subtitle={`${c.n_games.toLocaleString()} decisive games, walk-forward (no lookahead)`}>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <StatTile label="Vegas Brier" value={c.vegas.brier.toFixed(4)} tone="good" />
                <StatTile label="ML model Brier" value={c.ml ? c.ml.brier.toFixed(4) : "—"} />
                <StatTile label="Monte Carlo Brier" value={c.montecarlo.brier.toFixed(4)} tone="bad" />
                <StatTile
                  label="Statistically significant?"
                  value={c.mc_vs_vegas_bootstrap.vegas_significantly_better ? "Vegas wins" : c.mc_vs_vegas_bootstrap.mc_significantly_better ? "MC wins" : "No"}
                  tone={c.mc_vs_vegas_bootstrap.vegas_significantly_better ? "good" : "default"}
                />
              </div>
              <p className="text-xs text-slate-500">
                Lower Brier is better. Bootstrap 95% CI on the MC-vs-Vegas Brier difference: [{c.mc_vs_vegas_bootstrap.ci_low.toFixed(4)},{" "}
                {c.mc_vs_vegas_bootstrap.ci_high.toFixed(4)}]. The bar is "does the CI exclude zero", not which point estimate looks bigger.
              </p>
              {c.ml_vs_mc_bootstrap && (
                <p className="text-xs text-slate-500 mt-2">
                  <strong>The comparison that drives the season plan:</strong> future weeks have no posted line, so the ML model's real
                  competition is Monte Carlo, not Vegas.{" "}
                  {c.ml_vs_mc_bootstrap.ml_significantly_better
                    ? "The ML model wins that matchup with the CI excluding zero, which is why weeks beyond this one blend the two rather than using Monte Carlo alone."
                    : c.ml_vs_mc_bootstrap.mc_significantly_better
                      ? "Monte Carlo currently wins that matchup — worth re-checking before trusting ML-weighted future weeks."
                      : "Neither separates from the other yet on this sample."}{" "}
                  CI [{c.ml_vs_mc_bootstrap.ci_low.toFixed(4)}, {c.ml_vs_mc_bootstrap.ci_high.toFixed(4)}].
                </p>
              )}
            </Card>

            <Card title="Reliability diagram" subtitle="Predicted probability vs. what actually happened, by decile — the closer to the diagonal, the better calibrated">
              <ReliabilityChart calibration={c} />
            </Card>

            <Card title="High-confidence band (0.85–0.99)" subtitle="Exactly where survivor picks live — a model can look fine overall and still be overconfident here">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <ConfSliceTile label="Vegas" slice={c.vegas.high_confidence_slice} color={VEGAS_COLOR} />
                <ConfSliceTile label="ML model" slice={c.ml?.high_confidence_slice ?? null} color={ML_COLOR} />
                <ConfSliceTile label="Monte Carlo" slice={c.montecarlo.high_confidence_slice} color={MC_COLOR} />
              </div>
            </Card>

            <Card title="Brier score by week bucket" subtitle="Vegas is expected to dominate early weeks; the gap should narrow later">
              <WeekBucketChart calibration={c} />
            </Card>
          </>
        )}
      </AsyncView>

      <AsyncView state={backtest}>
        {(b) => (
          <Card title="Backtest: real seasons, 2010–2025" subtitle="Weeks survived per method, honest vs. real-lookahead ('cheating') variants">
            <BacktestChart summary={b.summary} />
          </Card>
        )}
      </AsyncView>
    </div>
  );
}

function ReliabilityChart({ calibration }: { calibration: Calibration }) {
  const vegasPoints = calibration.vegas.reliability.map((b) => ({ x: b.mean_predicted, y: b.empirical_rate, n: b.n }));
  const mcPoints = calibration.montecarlo.reliability.map((b) => ({ x: b.mean_predicted, y: b.empirical_rate, n: b.n }));
  const mlPoints = calibration.ml?.reliability.map((b) => ({ x: b.mean_predicted, y: b.empirical_rate, n: b.n })) ?? [];
  const diagonal = [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ];
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="x" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="Predicted" />
        <YAxis type="number" dataKey="y" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} name="Actual" />
        <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
        <Legend />
        <Line data={diagonal} dataKey="y" stroke="#cbd5e1" strokeDasharray="4 4" dot={false} activeDot={false} name="Perfect calibration" />
        <Scatter data={vegasPoints} fill={VEGAS_COLOR} name="Vegas" />
        <Scatter data={mcPoints} fill={MC_COLOR} name="Monte Carlo" />
        {mlPoints.length > 0 && <Scatter data={mlPoints} fill={ML_COLOR} name="ML model" />}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function ConfSliceTile({ label, slice, color }: { label: string; slice: ConfidenceSlice | null | undefined; color: string }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="text-sm font-medium mb-1" style={{ color }}>
        {label}
      </div>
      {!slice ? (
        <div className="text-sm text-slate-400">Not in this export</div>
      ) : slice.n === 0 ? (
        <div className="text-sm text-slate-400">No predictions in this band</div>
      ) : (
        <div className="text-sm text-slate-600 space-y-0.5">
          <div>n = {slice.n}</div>
          <div>predicted: {slice.mean_predicted !== null ? `${(slice.mean_predicted * 100).toFixed(1)}%` : "—"}</div>
          <div>actual: {slice.empirical_win_rate !== null ? `${(slice.empirical_win_rate * 100).toFixed(1)}%` : "—"}</div>
        </div>
      )}
    </div>
  );
}

function WeekBucketChart({ calibration }: { calibration: Calibration }) {
  const buckets = ["wk1-4", "wk5-9", "wk10-14", "wk15+"];
  const data = buckets.map((b) => ({
    bucket: b,
    Vegas: calibration.vegas.by_week_bucket[b]?.brier ?? null,
    "Monte Carlo": calibration.montecarlo.by_week_bucket[b]?.brier ?? null,
    "ML model": calibration.ml?.by_week_bucket[b]?.brier ?? null,
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="bucket" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="Vegas" fill={VEGAS_COLOR} radius={[4, 4, 0, 0]} />
        <Bar dataKey="ML model" fill={ML_COLOR} radius={[4, 4, 0, 0]} />
        <Bar dataKey="Monte Carlo" fill={MC_COLOR} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// Recharts drops X-axis tick labels that would overlap, silently. Six
// entries of "montecarlo cheating" length exceed that threshold where four
// did not, so the labels are shortened here and `interval={0}` below forces
// every one to render rather than letting the chart quietly hide some.
const METHOD_LABELS: Record<string, string> = {
  vegas: "Vegas",
  montecarlo: "Monte Carlo",
  ml: "ML model",
  blend: "Blend",
  blend3: "Blend (3-way)",
};

function backtestLabel(key: string): string {
  const separator = key.lastIndexOf("_");
  if (separator < 0) return key;
  const method = key.slice(0, separator);
  const variant = key.slice(separator + 1);
  return `${METHOD_LABELS[method] ?? method}\n${variant}`;
}

function BacktestTick({ x, y, payload }: { x?: number; y?: number; payload?: { value: string } }) {
  const [method, variant] = (payload?.value ?? "").split("\n");
  return (
    <g transform={`translate(${x ?? 0},${y ?? 0})`}>
      <text x={0} y={0} dy={12} textAnchor="middle" fontSize={11} fill="#475569">
        {method}
      </text>
      <text x={0} y={0} dy={26} textAnchor="middle" fontSize={10} fill="#94a3b8">
        {variant}
      </text>
    </g>
  );
}

function BacktestChart({ summary }: { summary: Record<string, BacktestMethodSummary> }) {
  const data = Object.entries(summary)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, s]) => ({
      key: backtestLabel(key),
      "Median weeks survived": s.median_weeks_survived,
      "Mean weeks survived": s.mean_weeks_survived,
    }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="key" tick={<BacktestTick />} interval={0} height={44} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="Median weeks survived" fill={VEGAS_COLOR} radius={[4, 4, 0, 0]} />
        <Bar dataKey="Mean weeks survived" fill="#93c5fd" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
