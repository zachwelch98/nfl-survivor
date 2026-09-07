import { useCallback, useEffect, useRef, useState } from "react";
import { Card } from "./Card";
import {
  PipelineApiError,
  getPipelineJob,
  listPipelineJobs,
  runPipelineCommand,
} from "../data/pipeline";
import type { PipelineJob, PipelineTarget } from "../data/pipeline";

interface CommandDef {
  target: PipelineTarget;
  label: string;
  description: string;
  // Whether this target should get --season/--week/--used (see Makefile).
  needsPickArgs?: boolean;
}

const COMMANDS: CommandDef[] = [
  {
    target: "refresh",
    label: "Refresh data",
    description: "Pull the latest nflverse data (lines, results, injuries) into the local cache.",
  },
  {
    target: "recommend",
    label: "Recommend",
    description: "Print this week's pick, the odds board, and the full season plan to the server console.",
    needsPickArgs: true,
  },
  {
    target: "export",
    label: "Export JSON",
    description: "Regenerate the JSON this dashboard reads (fast).",
    needsPickArgs: true,
  },
  {
    target: "export-full",
    label: "Export (full)",
    description: "Also refresh calibration, backtest, and simulation JSON (~90s).",
    needsPickArgs: true,
  },
  {
    target: "backtest",
    label: "Backtest",
    description: "Replay 2010–2025 for both models.",
  },
  {
    target: "test",
    label: "Run tests",
    description: "Run the Python test suite (177 tests).",
  },
];

const POLL_MS = 1200;

export function CommandRunner({
  season,
  week,
  usedTeams,
  onSuccess,
}: {
  season: number;
  week: number;
  usedTeams: string[];
  onSuccess?: (target: PipelineTarget) => void;
}) {
  const [job, setJob] = useState<PipelineJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const logRef = useRef<HTMLPreElement | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollJob = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = window.setInterval(() => {
        getPipelineJob(id)
          .then((j) => {
            setJob(j);
            if (j.status !== "running") {
              stopPolling();
              if (j.status === "success") onSuccess?.(j.target);
            }
          })
          .catch((err: unknown) => {
            stopPolling();
            setError(err instanceof Error ? err.message : String(err));
          });
      }, POLL_MS);
    },
    [onSuccess, stopPolling],
  );

  // Resume watching a job that's already running server-side -- e.g. this
  // tab reloaded, or a different tab started it -- rather than assuming idle.
  useEffect(() => {
    listPipelineJobs()
      .then(({ jobs, runningId }) => {
        if (!runningId) return;
        const running = jobs.find((j) => j.id === runningId);
        if (running) {
          setJob(running);
          pollJob(runningId);
        }
      })
      .catch(() => {
        // Pipeline API not reachable (e.g. a production build, which doesn't
        // carry this dev-only middleware) -- buttons will surface that on click.
      });
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (job?.status === "running" && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [job?.output, job?.status]);

  const running = job?.status === "running";

  async function trigger(cmd: CommandDef) {
    setError(null);
    try {
      const started = await runPipelineCommand(
        cmd.target,
        cmd.needsPickArgs ? { season, week, used: usedTeams.join(",") } : {},
      );
      setJob(started);
      pollJob(started.id);
    } catch (err) {
      if (err instanceof PipelineApiError && err.status === 409) {
        const runningJob = (err.body as { job?: PipelineJob } | null)?.job;
        if (runningJob) {
          setJob(runningJob);
          pollJob(runningJob.id);
        }
        setError(`"${cmd.label}" can't start — a command is already running.`);
        return;
      }
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <Card title="Pipeline controls" subtitle="Trigger the backend make commands without a terminal">
      <div className="flex flex-wrap gap-2">
        {COMMANDS.map((cmd) => {
          const isThisRunning = running && job?.target === cmd.target;
          return (
            <button
              key={cmd.target}
              type="button"
              title={cmd.description}
              disabled={running}
              onClick={() => void trigger(cmd)}
              className="rounded-md border border-slate-200 px-3 py-1.5 text-sm font-medium text-ink hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {isThisRunning ? "Running…" : cmd.label}
            </button>
          );
        })}
      </div>

      {error && <p className="mt-3 text-sm text-loss">{error}</p>}

      {job && (
        <div className="mt-4">
          <div className="flex items-center gap-2 text-xs flex-wrap">
            <StatusBadge status={job.status} />
            <span className="text-slate-500 font-mono">make {job.args.join(" ")}</span>
            {job.finishedAt && (
              <span className="text-slate-400">({((job.finishedAt - job.startedAt) / 1000).toFixed(1)}s)</span>
            )}
          </div>
          <pre
            ref={logRef}
            className="mt-2 max-h-64 overflow-auto rounded-lg bg-slate-950 text-slate-100 text-xs p-3 whitespace-pre-wrap leading-relaxed"
          >
            {job.output || "…"}
          </pre>
        </div>
      )}
    </Card>
  );
}

function StatusBadge({ status }: { status: PipelineJob["status"] }) {
  const toneClass =
    status === "running" ? "bg-tie/10 text-tie" : status === "success" ? "bg-win/10 text-win" : "bg-loss/10 text-loss";
  return <span className={`rounded-full px-2 py-0.5 font-medium ${toneClass}`}>{status}</span>;
}
