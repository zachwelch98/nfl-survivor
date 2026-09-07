// Dev-only Vite middleware exposing the project's `make <target>` commands
// over HTTP, so the Dashboard can trigger the backend pipeline (refresh,
// recommend, export, backtest, test) without a terminal.
//
// Scoped deliberately for a local, single-user tool: dev-server-only (this
// plugin hooks `configureServer`, so it's absent from `vite build`/`vite
// preview`), one job running at a time, and every target is checked against
// a fixed allowlist below -- this never shells out to a caller-supplied
// command string.

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";

const ALLOWED_TARGETS = ["refresh", "recommend", "export", "export-full", "backtest", "test"] as const;
type Target = (typeof ALLOWED_TARGETS)[number];

function isAllowedTarget(v: string): v is Target {
  return (ALLOWED_TARGETS as readonly string[]).includes(v);
}

const MAX_OUTPUT_CHARS = 300_000;
const MAX_FINISHED_JOBS_KEPT = 25;

export interface PipelineJob {
  id: string;
  target: Target;
  args: string[];
  status: "running" | "success" | "error";
  output: string;
  exitCode: number | null;
  startedAt: number;
  finishedAt: number | null;
}

function isValidSeason(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1999 && v <= 2100;
}
function isValidWeek(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v >= 1 && v <= 22;
}
function isValidUsed(v: unknown): v is string {
  return typeof v === "string" && (v === "" || /^[A-Z]{2,4}(,[A-Z]{2,4})*$/.test(v));
}

// Builds the `make` argv for a target. Values become separate argv entries
// (spawn is called without a shell), so this is not shell interpolation --
// the validation here is a usability guard (catch a bad request with a
// clear 400) rather than the injection defense; `isAllowedTarget` plus
// `shell: false` is what actually keeps this safe.
function buildArgs(target: Target, body: Record<string, unknown>): string[] {
  const args: string[] = [target];
  if (target === "recommend" || target === "export" || target === "export-full") {
    const season = body.season ?? 2026;
    const week = body.week ?? 1;
    const used = body.used ?? "";
    if (!isValidSeason(season)) throw new Error(`invalid season: ${JSON.stringify(season)}`);
    if (!isValidWeek(week)) throw new Error(`invalid week: ${JSON.stringify(week)}`);
    if (!isValidUsed(used)) {
      throw new Error(`invalid used (expected comma-separated team codes like "KC,BUF"): ${JSON.stringify(used)}`);
    }
    args.push(`SEASON=${season}`, `WEEK=${week}`, `USED=${used}`);
  }
  return args;
}

function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk: Buffer) => {
      data += chunk.toString();
      if (data.length > 1_000_000) {
        reject(new Error("request body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch {
        reject(new Error("invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

/** Registers the `/api/pipeline` dev-server API. `repoRoot` is where `make` runs. */
export function pipelineApi(repoRoot: string): Plugin {
  const jobs = new Map<string, PipelineJob>();
  let runningId: string | null = null;

  function pruneFinishedJobs() {
    if (jobs.size <= MAX_FINISHED_JOBS_KEPT) return;
    const finished = [...jobs.values()].filter((j) => j.status !== "running").sort((a, b) => a.startedAt - b.startedAt);
    while (jobs.size > MAX_FINISHED_JOBS_KEPT && finished.length) {
      jobs.delete(finished.shift()!.id);
    }
  }

  function appendOutput(job: PipelineJob, chunk: string) {
    job.output += chunk;
    if (job.output.length > MAX_OUTPUT_CHARS) {
      job.output = `…(truncated)…\n${job.output.slice(-MAX_OUTPUT_CHARS)}`;
    }
  }

  function startJob(target: Target, args: string[]): PipelineJob {
    const id = randomUUID();
    const job: PipelineJob = {
      id,
      target,
      args,
      status: "running",
      output: "",
      exitCode: null,
      startedAt: Date.now(),
      finishedAt: null,
    };
    jobs.set(id, job);
    runningId = id;

    const child = spawn("make", args, { cwd: repoRoot, env: process.env });
    child.stdout.on("data", (d: Buffer) => appendOutput(job, d.toString()));
    child.stderr.on("data", (d: Buffer) => appendOutput(job, d.toString()));
    child.on("error", (err) => {
      appendOutput(job, `\n[failed to launch make] ${err.message}\n`);
      job.status = "error";
      job.finishedAt = Date.now();
      if (runningId === id) runningId = null;
    });
    child.on("close", (code) => {
      job.exitCode = code;
      job.status = code === 0 ? "success" : "error";
      job.finishedAt = Date.now();
      if (runningId === id) runningId = null;
      pruneFinishedJobs();
    });

    return job;
  }

  return {
    name: "survivor-pipeline-api",
    configureServer(server) {
      server.middlewares.use("/api/pipeline", (req, res) => {
        void (async () => {
          try {
            const url = new URL(req.url ?? "/", "http://localhost");
            const parts = url.pathname.split("/").filter(Boolean);

            if (req.method === "GET" && parts.length === 0) {
              return sendJson(res, 200, { targets: ALLOWED_TARGETS, runningId });
            }

            if (req.method === "GET" && parts[0] === "jobs" && parts.length === 1) {
              const list = [...jobs.values()].sort((a, b) => b.startedAt - a.startedAt);
              return sendJson(res, 200, { jobs: list, runningId });
            }

            if (req.method === "GET" && parts[0] === "jobs" && parts.length === 2) {
              const job = jobs.get(parts[1]);
              if (!job) return sendJson(res, 404, { error: "job not found" });
              return sendJson(res, 200, job);
            }

            if (req.method === "POST" && parts[0] === "run" && parts.length === 1) {
              if (runningId) {
                const running = jobs.get(runningId);
                return sendJson(res, 409, {
                  error: "a pipeline command is already running",
                  job: running ?? null,
                });
              }
              const body = await readJsonBody(req);
              const target = body.target;
              if (typeof target !== "string" || !isAllowedTarget(target)) {
                return sendJson(res, 400, { error: `target must be one of: ${ALLOWED_TARGETS.join(", ")}` });
              }
              let args: string[];
              try {
                args = buildArgs(target, body);
              } catch (err) {
                return sendJson(res, 400, { error: err instanceof Error ? err.message : String(err) });
              }
              const job = startJob(target, args);
              return sendJson(res, 202, job);
            }

            sendJson(res, 404, { error: "not found" });
          } catch (err) {
            sendJson(res, 500, { error: err instanceof Error ? err.message : String(err) });
          }
        })();
      });
    },
  };
}
