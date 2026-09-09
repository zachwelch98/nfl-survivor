.PHONY: refresh recommend export export-full backtest test test-web web

SEASON ?= 2026
WEEK ?= 1
USED ?=

# The editable install's console script (`survivor`) depends on a .pth file
# that this Python 3.14 venv doesn't reliably process at startup — invoking
# via `-m` with PYTHONPATH set explicitly sidesteps that rather than
# depending on it.
PY := PYTHONPATH=backend .venv/bin/python -m survivor.cli

refresh:
	$(PY) refresh --pbp-seasons 2024,2025

recommend:
	$(PY) recommend --season $(SEASON) --week $(WEEK) --used "$(USED)"

export:
	$(PY) export --season $(SEASON) --week $(WEEK) --used "$(USED)"

export-full:
	$(PY) export --season $(SEASON) --week $(WEEK) --used "$(USED)" --with-analysis

backtest:
	$(PY) backtest --start 2010 --end 2025

test:
	.venv/bin/python -m pytest tests/ -q

# The browser-side assignment solver, checked against fixtures that `make test`
# generates from scipy. Run `make test` first if those fixtures are stale.
test-web:
	cd frontend && npm run test

web:
	cd frontend && npm run dev
