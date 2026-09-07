# NFL Survivor Pool ML Advisor — Frontend

Vite + React + TypeScript + Tailwind v4 + Recharts. Reads the JSON bundle the
Python pipeline writes to `public/data/` — see `src/data/source.ts` for the
one place that boundary is crossed (swapping static files for a live API
later is a change to that file only).

## Setup

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-checks (tsc -b) then builds to dist/
```

## Structure

- `src/data/types.ts` — TypeScript types mirroring
  `backend/survivor/export/json_bundle.py`'s output exactly (hand-kept in
  sync, not generated — see that file's docstring for the reasoning).
- `src/data/source.ts` — the only place views call `fetch`.
- `src/data/storage.ts` — localStorage persistence for picks/outcomes
  (zod-validated on read).
- `src/views/` — the six views: Dashboard, Odds Board, Season Plan, My
  Picks, Analysis, Simulation.
- `src/components/` — shared presentational pieces (Card, ProbBar, etc).

## Data

`public/data/*.json` is committed (not gitignored) — it's the live output of
`make export`, run against real nflverse data, not disposable mock data. Run
`make export` (or `make export-full` for the slower calibration/backtest/
simulation refresh) from the repo root to regenerate it with fresh numbers.
