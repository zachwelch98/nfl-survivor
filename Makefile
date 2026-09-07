.PHONY: refresh recommend export export-full backtest test web

SEASON ?= 2026
WEEK ?= 1
USED ?=

refresh:
	.venv/bin/survivor refresh --pbp-seasons 2024,2025

recommend:
	.venv/bin/survivor recommend --season $(SEASON) --week $(WEEK) --used "$(USED)"

export:
	.venv/bin/survivor export --season $(SEASON) --week $(WEEK) --used "$(USED)"

export-full:
	.venv/bin/survivor export --season $(SEASON) --week $(WEEK) --used "$(USED)" --with-analysis

backtest:
	.venv/bin/survivor backtest --start 2010 --end 2025

test:
	.venv/bin/python -m pytest tests/ -q

web:
	cd frontend && npm run dev
