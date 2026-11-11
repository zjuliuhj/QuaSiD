#!/usr/bin/env bash
set -euo pipefail

# Queue forecasting experiments across datasets, history lengths, and horizons.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

DATASETS=${DATASETS:-"be de electricity etth1 etth2 ettm1 ettm2 exchange fr stocks weather"}
HISTORY_LENGTHS=${HISTORY_LENGTHS:-"96"}
PRED_LENGTHS=${PRED_LENGTHS:-"96 192 336 720"}
CONDITION_DOMAIN=${CONDITION_DOMAIN:-frequency}
MODEL_NAME=${MODEL_NAME:-quasid_sde}
BACKBONE=${BACKBONE:-dit_solver_v5}
USE_PMI_CORR=${USE_PMI_CORR:-true}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_EVAL=${RUN_EVAL:-true}
TENSORBOARD=${TENSORBOARD:-true}
MILESTONE=${MILESTONE:-5}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-"0"}
LOG_ROOT=${LOG_ROOT:-logs/forecasting_queue}
FORECAST_SAMPLES=${FORECAST_SAMPLES:-20}
ENSEMBLE_CHUNK_SIZE=${ENSEMBLE_CHUNK_SIZE:-4}
FORECAST_EVAL_SAMPLE_LIMIT=${FORECAST_EVAL_SAMPLE_LIMIT:-2000}

TASK_FILE=$(mktemp "${TMPDIR:-/tmp}/quasid_forecasting_tasks.XXXXXX")
trap 'rm -f "${TASK_FILE}"' EXIT

for dataset in ${DATASETS}; do
  config_file="config/forecasting/${dataset}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${config_file}" >&2
    continue
  fi
  for history in ${HISTORY_LENGTHS}; do
    for horizon in ${PRED_LENGTHS}; do
      echo "MODE=forecast DATASET=${dataset} HISTORY=${history} HORIZON=${horizon} CONDITION_DOMAIN=${CONDITION_DOMAIN} MODEL_NAME=${MODEL_NAME} BACKBONE=${BACKBONE} USE_PMI_CORR=${USE_PMI_CORR} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL} TENSORBOARD=${TENSORBOARD} MILESTONE=${MILESTONE} PYTHON_BIN=${PYTHON_BIN}" >> "${TASK_FILE}"
    done
  done
done

TASK_FILE="${TASK_FILE}" \
GPU_IDS="${GPU_IDS}" \
LOG_ROOT="${LOG_ROOT}" \
TASK_RUNNER="bash scripts/experiments/train_eval.sh --num_samples ${FORECAST_SAMPLES} --ensemble_chunk_size ${ENSEMBLE_CHUNK_SIZE} --eval_sample_limit ${FORECAST_EVAL_SAMPLE_LIMIT}" \
bash scripts/experiments/schedule_gpu_queue.sh
