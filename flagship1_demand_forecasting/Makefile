PYTHON ?= python3
UV ?= uv

define RUN
if command -v $(UV) >/dev/null 2>&1; then $(UV) run --extra dev $(1); else $(1); fi
endef

install:
	if command -v $(UV) >/dev/null 2>&1; then $(UV) sync --extra dev; else $(PYTHON) -m pip install -e '.[dev]'; fi

download-data:
	$(call RUN,$(PYTHON) scripts/download_data.py)

train: download-data
	$(call RUN,$(PYTHON) -m demand_forecasting.train)

test:
	$(call RUN,$(PYTHON) -m pytest -q)

lint:
	$(call RUN,$(PYTHON) -m ruff check .)

run-api:
	$(call RUN,$(PYTHON) -m uvicorn demand_forecasting.api:app --host 0.0.0.0 --port 8000)

docker-up:
	docker compose up --build
