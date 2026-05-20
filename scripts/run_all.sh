#!/usr/bin/env bash
set -euo pipefail
python -m loto_forecast.cli db-init
python -m loto_forecast.cli catalog-import --library neuralforecast
python -m loto_forecast.cli train --model AutoNHITS --h 28 --params-json '{"num_samples":10,"seed":1,"backend":"optuna"}'
