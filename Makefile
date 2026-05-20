UV ?= uv
PYTHON_VERSION ?= 3.11
UV_SYNC_FLAGS ?= --extra dev
UV_LINK_MODE ?= copy
LOTO_UV_ENV_MODE ?= static

.PHONY: venv install sync check test lint format format-fix mypy bandit db-init db-init-apply train catalog-import grid-create grid-run build-exog dashboard screenshots feature-job feature-job-apply automation-preview automation-install clean

venv:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) venv --python $(PYTHON_VERSION)

install: sync

sync:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync $(UV_SYNC_FLAGS)

check:
	./scripts/verify_static.sh

test:
	PYTHONPATH=src $(UV) run pytest tests/unit -v --tb=short --no-cov

lint:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync --extra dev
	$(UV) run --no-sync ruff check src tests --no-fix

format:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync --extra dev
	$(UV) run --no-sync ruff format src tests --check

format-fix:
	./scripts/fix_style.sh

mypy:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync --extra dev
	$(UV) run --no-sync mypy src/loto_forecast --ignore-missing-imports

bandit:
	UV_LINK_MODE=$(UV_LINK_MODE) $(UV) sync --extra dev
	$(UV) run --no-sync bandit -r src/loto_forecast -c pyproject.toml

db-init:
	@echo "SAFE DEFAULT: dry-run only. No SQL will be executed."
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli db-init --dry-run

db-init-apply:
	@if [ "$$LOTO_ALLOW_DB_INIT" != "1" ]; then \
		echo "Refusing to run db-init. Set LOTO_ALLOW_DB_INIT=1 after backup confirmation."; \
		exit 2; \
	fi
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli db-init --yes-i-understand-db-init-may-write

train:
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli train --model AutoNHITS --h 28 --params-json '{"num_samples":10,"seed":1}'

catalog-import:
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli catalog-import --library neuralforecast

grid-create:
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli grid-create --grid-id nf_grid_001 --adapter neuralforecast_auto --model AutoNHITS --h 28 --param-space-json '{"num_samples":[10,20],"seed":[1,2],"backend":["optuna"]}'

grid-run:
	@if [ "$$LOTO_ALLOW_GRID_RUN" != "1" ]; then \
		echo "Refusing to run grid-run by default. Set LOTO_ALLOW_GRID_RUN=1 after cost/time/DB-write confirmation."; \
		exit 2; \
	fi
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli grid-run --grid-id nf_grid_001

build-exog:
	PYTHONPATH=src $(UV) run python -m loto_forecast.cli build-exog --target-schema exog --target-table loto_y_ts_exog --group-cols 'loto,unique_id,ts_type' --parallel-workers 4

dashboard:
	PYTHONPATH=src $(UV) run streamlit run src/loto_forecast/api/streamlit/operations_dashboard.py


screenshots:
	LOTO_UV_ENV_MODE=browser LOTO_UV_CLEAR_VENV=0 ./scripts/capture_app_screenshots.sh --max-clicks 40 --max-depth 2

feature-job:
	./scripts/run_dataset_feature_table_job.sh --source-schema dataset --source-table loto_y_ts_unified --target-schema exog --target-table nf_feature_table_auto --limit 5000

feature-job-apply:
	@if [ "$$LOTO_ALLOW_FEATURE_DB_WRITE" != "1" ]; then \
		echo "Refusing to write feature table. Set LOTO_ALLOW_FEATURE_DB_WRITE=1 after DB backup/target confirmation."; \
		exit 2; \
	fi
	./scripts/run_dataset_feature_table_job.sh --source-schema dataset --source-table loto_y_ts_unified --target-schema exog --target-table nf_feature_table_auto --limit 5000 --yes-write

automation-preview:
	./scripts/install_wsl_automation.sh --all

automation-install:
	@if [ "$$LOTO_ALLOW_AUTOMATION_INSTALL" != "1" ]; then \
		echo "Refusing to install crontab. Set LOTO_ALLOW_AUTOMATION_INSTALL=1 after reviewing automation-preview."; \
		exit 2; \
	fi
	./scripts/install_wsl_automation.sh --install --all


clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
