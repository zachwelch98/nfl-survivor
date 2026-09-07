import { AsyncView } from "../components/AsyncView";
import { Card, StatTile } from "../components/Card";
import { useAsync } from "../lib/useAsync";
import { getSeasonMeta } from "../data/source";
import { useUserState } from "../lib/useUserState";
import { survivalRecord } from "../data/storage";
import { TEAM_NAMES, teamName } from "../lib/teams";
import type { UserPick } from "../data/types";

const ALL_TEAMS = Object.keys(TEAM_NAMES).sort();
const WEEKS = Array.from({ length: 18 }, (_, i) => i + 1);

export default function MyPicks() {
  const meta = useAsync(() => getSeasonMeta(), []);
  const { state, setPick, clearPick } = useUserState();

  return (
    <AsyncView state={meta}>
      {(m) => {
        const record = survivalRecord(state, m.season);
        const picksByWeek = new Map(state.picks.filter((p) => p.season === m.season).map((p) => [p.week, p]));

        return (
          <div className="space-y-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">My Picks</h1>
              <p className="text-sm text-slate-500 mt-1">Record your pick and its outcome each week. A team already used anywhere this season is unavailable everywhere else.</p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatTile label="Wins" value={record.wins} tone="good" />
              <StatTile label="Losses" value={record.losses} tone={record.losses > 0 ? "bad" : "default"} />
              <StatTile label="Ties" value={record.ties} />
              <StatTile
                label="Status"
                value={record.eliminatedWeek ? `Out (Wk ${record.eliminatedWeek})` : "Alive"}
                tone={record.eliminatedWeek ? "bad" : "good"}
              />
            </div>

            <Card>
              <div className="divide-y divide-slate-100">
                {WEEKS.map((week) => {
                  const pick = picksByWeek.get(week);
                  const usedElsewhere = new Set(
                    state.picks.filter((p) => p.season === m.season && p.week !== week).map((p) => p.team),
                  );
                  return (
                    <WeekRow
                      key={week}
                      week={week}
                      pick={pick}
                      usedElsewhere={usedElsewhere}
                      onSetTeam={(team) => setPick({ season: m.season, week, team, outcome: pick?.outcome ?? "pending" })}
                      onSetOutcome={(outcome) => pick && setPick({ ...pick, outcome })}
                      onClear={() => clearPick(m.season, week)}
                    />
                  );
                })}
              </div>
            </Card>
          </div>
        );
      }}
    </AsyncView>
  );
}

function WeekRow({
  week,
  pick,
  usedElsewhere,
  onSetTeam,
  onSetOutcome,
  onClear,
}: {
  week: number;
  pick: UserPick | undefined;
  usedElsewhere: Set<string>;
  onSetTeam: (team: string) => void;
  onSetOutcome: (outcome: UserPick["outcome"]) => void;
  onClear: () => void;
}) {
  const outcomeStyle: Record<UserPick["outcome"], string> = {
    pending: "bg-slate-100 text-slate-500",
    win: "bg-emerald-100 text-emerald-700",
    loss: "bg-red-100 text-red-700",
    tie: "bg-amber-100 text-amber-700",
  };

  return (
    <div className="flex items-center gap-3 py-2.5 flex-wrap">
      <div className="w-16 text-sm font-medium text-slate-500">Week {week}</div>
      <select
        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm min-w-40"
        value={pick?.team ?? ""}
        onChange={(e) => (e.target.value ? onSetTeam(e.target.value) : onClear())}
      >
        <option value="">— pick a team —</option>
        {ALL_TEAMS.map((t) => (
          <option key={t} value={t} disabled={usedElsewhere.has(t)}>
            {t} — {teamName(t)}
            {usedElsewhere.has(t) ? " (used)" : ""}
          </option>
        ))}
      </select>

      {pick && (
        <div className="flex items-center gap-1.5 ml-auto">
          {(["pending", "win", "loss", "tie"] as const).map((o) => (
            <button
              key={o}
              onClick={() => onSetOutcome(o)}
              className={`px-2 py-1 rounded-md text-xs font-medium capitalize transition-colors ${
                pick.outcome === o ? outcomeStyle[o] + " ring-1 ring-inset ring-current" : "text-slate-400 hover:bg-slate-100"
              }`}
            >
              {o}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
