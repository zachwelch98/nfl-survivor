import { AsyncView } from "../components/AsyncView";
import { Card } from "../components/Card";
import { ProbPill } from "../components/ProbBar";
import { useAsync } from "../lib/useAsync";
import { getSeasonPlan } from "../data/source";
import { teamName } from "../lib/teams";

export default function SeasonPlanView() {
  const plan = useAsync(() => getSeasonPlan(), []);
  return (
    <AsyncView state={plan}>
      {(p) => (
        <div className="space-y-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Season Plan</h1>
            <p className="text-sm text-slate-500 mt-1">
              The exact optimal pick sequence for the rest of the season — solved as a bipartite assignment problem, not
              picked week by week.
            </p>
          </div>

          <Card title="Recommended sequence" subtitle={p.total_survival_prob ? `Total survival probability across all ${p.recommended_sequence.length} weeks: ${(p.total_survival_prob * 100).toFixed(3)}%` : undefined}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-slate-400 border-b border-slate-200">
                    <th className="py-2 pr-3">Week</th>
                    <th className="py-2 px-3">Team</th>
                    <th className="py-2 pl-3">Survival probability</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {p.recommended_sequence.map((pick) => (
                    <tr key={pick.week}>
                      <td className="py-1.5 pr-3 text-slate-500">Week {pick.week}</td>
                      <td className="py-1.5 px-3 font-medium">
                        {pick.team} <span className="text-slate-400 font-normal">{teamName(pick.team)}</span>
                      </td>
                      <td className="py-1.5 pl-3">
                        <ProbPill value={pick.survival_prob} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {p.alternates.length > 1 && (
            <Card title="Alternate season paths" subtitle="The next-best complete plans, ranked by total survival probability">
              <div className="space-y-3">
                {p.alternates.map((alt, i) => (
                  <div key={i} className="rounded-lg border border-slate-200 p-3">
                    <div className="text-sm font-medium mb-2">
                      Plan #{i + 1} — {(alt.total_survival_prob * 100).toFixed(3)}% total survival
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {alt.picks.map((pick) => (
                        <span key={pick.week} className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-xs">
                          W{pick.week}: {pick.team}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </AsyncView>
  );
}
