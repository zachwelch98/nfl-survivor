# NFL Survivor Pool ML Advisor

A tool that recommends which team to pick each week in an NFL survivor pool —
optimized for survival across the *full season*, not just this week's win
probability.

## Status: complete

1. [x] Data layer: nflverse loaders, leakage gate (`AsOf`), stadium geography
2. [x] Method 1: Vegas-implied win probability (baseline)
3. [x] As-of feature store: team ratings, drive rates, situational adjustments
4. [x] Method 2: drive-level Monte Carlo game simulator (challenger)
5. [x] Model evaluation: Brier score, log loss, calibration
6. [x] Season-sequencing optimizer (exact assignment solve)
7. [x] Backtest harness (2010–2025, no-leakage + real-lookahead variants)
8. [x] Season & pool simulation
9. [x] JSON export for the frontend
10. [x] React app (Vite + TS + Tailwind + Recharts)

## The headline empirical finding

**Vegas closing lines beat the Monte Carlo simulator, decisively.** Walk-forward
on real 2022–2024 games (n=813-1084 depending on the run): Vegas Brier ≈0.21 vs.
Monte Carlo ≈0.25, bootstrap CI on the difference excludes zero. Worse, Monte
Carlo is specifically *overconfident* in the 0.85–0.99 win-probability band —
exactly where survivor picks live — predicting ~90% and delivering ~71–78%
actual. Blending the two doesn't help either. `config.yaml`'s
`model.primary_method: vegas` reflects this; see `evaluate/model_comparison.py`
and the Analysis view for the full evidence. This isn't a failure of the
project — it's the answer plan §3 asked for ("don't assume the answer"), and
building Method 2 rigorously is what let the finding be trusted at all.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 177 tests
```

Use `make recommend` / `make export` / `make backtest` (see Makefile) rather
than the installed `survivor` console script directly — on this Python 3.14
venv the editable install's `.pth` file isn't reliably processed by `site.py`
at interpreter startup, so the Makefile invokes `python -m survivor.cli` with
`PYTHONPATH=backend` set explicitly instead.

```bash
make recommend                          # this week's pick, current season
make export                             # write the JSON bundle (fast)
make export-full                        # also refresh calibration/backtest/simulation (~90s)
make backtest                           # replay 2010-2025 for both methods
make test                               # run the Python test suite
make web                                # cd frontend && npm run dev
```

`frontend/public/data/*.json` is committed as a live snapshot (not gitignored
mock data) — the pipeline was built before the frontend, so there was real
data to show by the time it existed.

## Key design decisions

- **No paid odds API.** nflverse's `games.csv` (via `nflreadpy`) already
  carries closing spreads (1999+) and moneylines (2010+), which is what both
  the Vegas baseline and the backtest harness use — zero cost, zero keys,
  and train/serve use the identical data source.
- **The leakage gate is one file.** `survivor.clock.AsOf` is the single
  enforcement point for "no lookahead" — every feature builder and backtest
  step must go through `AsOf.visible(df)` rather than filtering season/week
  by hand. See `backend/survivor/clock.py` and `tests/test_clock.py`.
- **The season optimizer is exact, not heuristic.** Maximizing survival
  probability across the season is equivalent to maximizing
  `Σ log p(pick_w)`, which is a bipartite assignment problem (weeks × teams)
  solvable exactly and instantly with `scipy.optimize.linear_sum_assignment`.
  `optimize/kbest.py` adds Murty's algorithm for the k best alternate plans
  and a cost-of-deviation figure for every team playable in a given week.
- **Future weeks never touch a real posted line.** Beyond the current week,
  probabilities come from Method 2's simulation with confidence decay toward
  a coin flip as the horizon grows (`optimize/projection.py`) — using a
  future week's actual closing line, even in a historical backtest, would be
  lookahead (that value bakes in information — injury news, sharp money —
  that didn't exist yet at decision time). The backtest harness also runs a
  deliberately-cheating variant that *does* use real closing lines throughout,
  purely to quantify what that lookahead would be worth.
- **The frontend is a thin, swappable read layer.** Every view fetches through
  `frontend/src/data/source.ts`; nothing else calls `fetch`. Swapping static
  JSON for a live API later is a one-file change.
- **The Dashboard can trigger the pipeline itself.** A dev-only Vite
  middleware (`frontend/vite-plugins/pipeline.ts`) exposes `/api/pipeline/*`
  and shells out to `make <target>` (allowlisted, one job at a time) so
  `make refresh|recommend|export|export-full|backtest|test` can be run from
  a button instead of a terminal — see `frontend/README.md`.

## Project layout

```
backend/survivor/
  clock.py            AsOf leakage gate
  config.py            typed config.yaml loader
  data/                 nflverse loaders, stadium geography
  features/            as-of team ratings, drive rates, situational adjustments
  models/               vegas.py, montecarlo.py, blend.py
  evaluate/             Brier/log-loss/calibration, walk-forward model comparison
  optimize/             exact assignment solver, k-best, cost-of-deviation, projection
  backtest/             replay real seasons week-by-week
  simulate/             strategy comparison (season.py), pool-field simulation (pool.py)
  export/               JSON bundle for the frontend
  cli.py                survivor refresh|recommend|export|backtest
frontend/               Vite + React + TS + Tailwind + Recharts
tests/                  177 tests, pytest
```
