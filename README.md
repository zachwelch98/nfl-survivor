# NFL Survivor Pool ML Advisor

A tool that recommends which team to pick each week in an NFL survivor pool —
optimized for survival across the *full season*, not just this week's win
probability. See the design rationale and architecture in the implementation
plan this project was built from.

## Status

Work in progress, building in this order:

1. [x] Data layer: nflverse loaders, leakage gate (`AsOf`), stadium geography
2. [ ] Method 1: Vegas-implied win probability (baseline)
3. [ ] As-of feature store: team ratings, drive rates, situational adjustments
4. [ ] Method 2: drive-level Monte Carlo game simulator (challenger)
5. [ ] Model evaluation: Brier score, log loss, calibration
6. [ ] Season-sequencing optimizer (exact assignment solve)
7. [ ] Backtest harness (2010–2025, no-leakage)
8. [ ] Season & pool simulation
9. [ ] JSON export for the frontend
10. [ ] React app (Vite + TS + Tailwind + Recharts)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Use `make recommend` / `make export` / `make backtest` (see Makefile) rather
than the installed `survivor` console script directly — on this Python 3.14
venv the editable install's `.pth` file isn't reliably processed by `site.py`
at interpreter startup, so the Makefile invokes `python -m survivor.cli` with
`PYTHONPATH=backend` set explicitly instead.

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
