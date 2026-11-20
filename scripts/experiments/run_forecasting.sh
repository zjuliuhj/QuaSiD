#!/usr/bin/env bash
set -euo pipefail

# Queue forecasting experiments across datasets, history lengths, and horizons.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

DATASETS=${DATASETS:-"etth1 etth2 ettm1 ettm2 weather be de exchange ili"}
HISTORY_LENGTHS=${HISTORY_LENGTHS:-168}
PRED_LENGTHS=${PRED_LENGTHS:-192}
ILI_PRED_LENGTHS=${ILI_PRED_LENGTHS:-36}
CONDITION_DOMAIN=${CONDITION_DOMAIN:-frequency}
MODEL_NAME=${MODEL_NAME:-quasid_cont}
BACKBONE=${BACKBONE:-dit_solver_v5}
USE_PMI_CORR=${USE_PMI_CORR:-true}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_EVAL=${RUN_EVAL:-true}
TENSORBOARD=${TENSORBOARD:-true}
MILESTONE=${MILESTONE:-best}
EVAL_SEEDS=${EVAL_SEEDS:-"2026"}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-"0"}
LOG_ROOT=${LOG_ROOT:-logs/forecasting_queue}
FORECAST_SAMPLES=${FORECAST_SAMPLES:-20}
ENSEMBLE_CHUNK_SIZE=${ENSEMBLE_CHUNK_SIZE:-4}
FORECAST_EVAL_SAMPLE_LIMIT=${FORECAST_EVAL_SAMPLE_LIMIT:-20000}

TASK_FILE=$(mktemp "${TMPDIR:-/tmp}/quasid_forecasting_tasks.XXXXXX")
trap 'rm -f "${TASK_FILE}"' EXIT

for dataset in ${DATASETS}; do
  config_file="config/forecasting/${dataset}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${config_file}" >&2
    continue
  fi
  history_lengths=${HISTORY_LENGTHS:-$(awk '$1 == "history_length:" {print $2; exit}' "${config_file}")}
  if [[ "${dataset}" == "ili" ]]; then
    pred_lengths=${ILI_PRED_LENGTHS}
  else
    pred_lengths=${PRED_LENGTHS}
  fi

  for model_name in ${MODEL_NAME}; do
    case "${model_name}" in
      quasid_cont|quasid_disc) ;;
      *)
        echo "MODEL_NAME entries must be quasid_cont or quasid_disc. Got ${model_name}" >&2
        exit 1
        ;;
    esac
    for history in ${history_lengths}; do
      for horizon in ${pred_lengths}; do
        echo "MODE=forecast DATASET=${dataset} HISTORY=${history} HORIZON=${horizon} CONDITION_DOMAIN=${CONDITION_DOMAIN} MODEL_NAME=${model_name} BACKBONE=${BACKBONE} USE_PMI_CORR=${USE_PMI_CORR} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL} TENSORBOARD=${TENSORBOARD} MILESTONE=${MILESTONE} EVAL_SEEDS=${EVAL_SEEDS} PYTHON_BIN=${PYTHON_BIN}" >> "${TASK_FILE}"
      done
    done
  done
done

TASK_FILE="${TASK_FILE}" \
GPU_IDS="${GPU_IDS}" \
LOG_ROOT="${LOG_ROOT}" \
TASK_RUNNER="bash scripts/experiments/train_eval.sh --num_samples ${FORECAST_SAMPLES} --ensemble_chunk_size ${ENSEMBLE_CHUNK_SIZE} --eval_sample_limit ${FORECAST_EVAL_SAMPLE_LIMIT}" \
bash scripts/experiments/schedule_gpu_queue.sh
