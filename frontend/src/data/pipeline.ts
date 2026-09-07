// Client for the dev-only pipeline API (see ../../vite-plugins/pipeline.ts)
// that lets the Dashboard trigger `make <target>` directly instead of a
// terminal. Same fetch-boundary discipline as source.ts: this is the only
// file that talks to /api/pipeline.

export type PipelineTarget = "refresh" | "recommend" | "export" | "export-full" | "backtest" | "test";

export interface PipelineJob {
  id: string;
  target: PipelineTarget;
  args: string[];
  status: "running" | "success" | "error";
  output: string;
  exitCode: number | null;
  startedAt: number;
  finishedAt: number | null;
}

export interface RunOptions {
  season?: number;
  week?: number;
  used?: string;
}

export class PipelineApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, message: string, body: unknown) {
    super(message);
    this.name = "PipelineApiError";
    this.status = status;
    this.body = body;
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/pipeline${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body: unknown = await res.json().catch(() => ({}));
  if (!res.ok) {
    const message = (body as { error?: string } | null)?.error ?? `${res.status} ${res.statusText}`;
    throw new PipelineApiError(res.status, message, body);
  }
  return body as T;
}

export function runPipelineCommand(target: PipelineTarget, opts: RunOptions = {}): Promise<PipelineJob> {
  return api<PipelineJob>("/run", { method: "POST", body: JSON.stringify({ target, ...opts }) });
}

export function getPipelineJob(id: string): Promise<PipelineJob> {
  return api<PipelineJob>(`/jobs/${id}`);
}

export function listPipelineJobs(): Promise<{ jobs: PipelineJob[]; runningId: string | null }> {
  return api(`/jobs`);
}
