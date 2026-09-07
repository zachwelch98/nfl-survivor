import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AsyncView } from "../components/AsyncView";
import { Card, StatTile } from "../components/Card";
import { useAsync } from "../lib/useAsync";
import { getSimulation } from "../data/source";

const STRATEGY_COLORS: Record<string, string> = {
  greedy: "#f59e0b",
  static: "#94a3b8",
  adaptive: "#2563eb",
  save_best: "#16a34a",
};

const STRATEGY_LABELS: Record<string, string> = {
  greedy: "Greedy (max win% each week)",
  static: "Static (plan fixed at week 1)",
  adaptive: "Adaptive (re-solved weekly)",
  save_best: "Save-your-best heuristic",
};

export default function SimulationView() {
  const simulation = useAsync(() => getSimulation(), []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Strategy Simulation</h1>
        <p className="text-sm text-slate-500 mt-1">
          Same underlying game probabilities, many simulated seasons, four decision rules compared by outcome.
        </p>
      </div>

      <AsyncView state={simulation}>
        {(sim) => {
          const entries = Object.entries(sim);
          const curveData = buildCurveData(sim);
          return (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {entries.map(([key, s]) => (
                  <Card key={key} title={STRATEGY_LABELS[key] ?? key}>
                    <div className="space-y-2">
                      <StatTile label="Median weeks survived" value={s.median_weeks_survived.toFixed(1)} />
                      <StatTile label="Mean weeks survived" value={s.mean_weeks_survived.toFixed(2)} />
                      <StatTile label="P(survive full season)" value={`${(s.p_survive_full_run * 100).toFixed(2)}%`} />
                    </div>
                  </Card>
                ))}
              </div>

              <Card title="Survival curve" subtitle="P(still alive) at each week, per strategy">
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={curveData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="week" label={{ value: "Week", position: "insideBottom", offset: -4 }} />
                    <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} labelFormatter={(w) => `Week ${w}`} />
                    <Legend formatter={(value) => STRATEGY_LABELS[value] ?? value} />
                    {entries.map(([key]) => (
                      <Line key={key} type="monotone" dataKey={key} stroke={STRATEGY_COLORS[key] ?? "#000"} strokeWidth={2} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Reading these results">
                <p className="text-sm text-slate-600 leading-relaxed">
                  The assignment optimizer is mathematically guaranteed to maximize the probability of surviving the{" "}
                  <em>entire</em> remaining season for a fixed set of probabilities — verified against brute force. But{" "}
                  <strong>adaptive</strong> re-solves each week against genuinely evolving information, and "mean weeks
                  survived" is a different objective than "P(survive everything)" — so greedy can occasionally look
                  competitive or better on that specific metric in any one real season, even though it has no such
                  guarantee. Static (committing to week 1's plan and never revising) is consistently the weakest here,
                  which is the expected result of ignoring new information as the season unfolds.
                </p>
              </Card>
            </>
          );
        }}
      </AsyncView>
    </div>
  );
}

function buildCurveData(sim: Awaited<ReturnType<typeof getSimulation>>) {
  const weeks = new Set<number>();
  for (const s of Object.values(sim)) {
    for (const w of Object.keys(s.survive_to_week_curve)) weeks.add(Number(w));
  }
  return Array.from(weeks)
    .sort((a, b) => a - b)
    .map((week) => {
      const row: Record<string, number> = { week };
      for (const [key, s] of Object.entries(sim)) {
        row[key] = s.survive_to_week_curve[String(week)] ?? s.survive_to_week_curve[week];
      }
      return row;
    });
}
