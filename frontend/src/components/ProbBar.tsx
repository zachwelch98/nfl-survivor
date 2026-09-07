// Small horizontal win-probability bar: home team's share fills from the
// left in accent blue, away team's share from the right in slate.
export function ProbBar({ homeProb, homeLabel, awayLabel }: { homeProb: number; homeLabel: string; awayLabel: string }) {
  const pct = Math.round(homeProb * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-500 mb-1">
        <span>{awayLabel}</span>
        <span>{homeLabel}</span>
      </div>
      <div className="h-2.5 rounded-full bg-slate-100 overflow-hidden flex">
        <div className="h-full bg-slate-400" style={{ width: `${100 - pct}%` }} />
        <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ProbPill({ value }: { value: number | null }) {
  if (value === null) return <span className="text-slate-300">—</span>;
  const pct = Math.round(value * 100);
  const tone = pct >= 65 ? "bg-emerald-50 text-emerald-700 border-emerald-200" : pct <= 40 ? "bg-red-50 text-red-700 border-red-200" : "bg-slate-50 text-slate-600 border-slate-200";
  return <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-xs font-medium tabular-nums ${tone}`}>{pct}%</span>;
}
