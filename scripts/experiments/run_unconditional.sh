#!/usr/bin/env bash
set -euo pipefail

# Queue unconditional generation experiments across models, datasets, and sequence lengths.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

DATASETS=${DATASETS:-"air etth1 sines stocks Quito"}
SEQ_LENGTHS=${SEQ_LENGTHS:-"64 128 256 512"}
MODEL_NAME=${MODEL_NAME:-"quasid_disc quasid_cont"}
BACKBONE=${BACKBONE:-dit_solver_v5}
USE_PMI_CORR=${USE_PMI_CORR:-true}
EMB_SIZES=${EMB_SIZES:-${EMB_SIZE:-256}}
NUM_LAYERS_LIST=${NUM_LAYERS_LIST:-${NUM_LAYERS:-4}}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_EVAL=${RUN_EVAL:-true}
TENSORBOARD=${TENSORBOARD:-true}
EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS:-true}
MILESTONE=${MILESTONE:-10}
EVAL_SEEDS=${EVAL_SEEDS:-"2021,2022,2023,2024,2025"}
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi
GPU_IDS=${GPU_IDS:-"0"}
LOG_ROOT=${LOG_ROOT:-logs/unconditional_queueQuaSiD_${BACKBONE}}

TASK_FILE=$(mktemp "${TMPDIR:-/tmp}/quasid_unconditional_tasks.XXXXXX")
trap 'rm -f "${TASK_FILE}"' EXIT

yaml_scalar() {
  local config_file=$1 key=$2
  awk -v key="${key}:" '$1 == key {print $2; exit}' "${config_file}"
}

for dataset in ${DATASETS}; do
  config_file="config/unconditional/${dataset}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${config_file}" >&2
    continue
  fi

  emb_sizes=${EMB_SIZES:-$(yaml_scalar "${config_file}" emb_size)}
  num_layers_values=${NUM_LAYERS_LIST:-$(yaml_scalar "${config_file}" num_layers)}
  if [[ -z "${emb_sizes}" || -z "${num_layers_values}" ]]; then
    echo "SKIP dataset=${dataset}: could not infer emb_size or num_layers from ${config_file}" >&2
    continue
  fi

  for model_name in ${MODEL_NAME}; do
    task_backbone=${BACKBONE}
    task_use_pmi_corr=${USE_PMI_CORR}
    case "${model_name}" in
      quasid_cont|quasid_disc) ;;
      *)
        echo "MODEL_NAME entries must be quasid_cont or quasid_disc. Got ${model_name}" >&2
        exit 1
        ;;
    esac
    for emb_size in ${emb_sizes}; do
      for num_layers in ${num_layers_values}; do
        for seq_length in ${SEQ_LENGTHS}; do
          echo "MODE=uncond DATASET=${dataset} SEQ_LENGTH=${seq_length} MODEL_NAME=${model_name} BACKBONE=${task_backbone} USE_PMI_CORR=${task_use_pmi_corr} EMB_SIZE=${emb_size} NUM_LAYERS=${num_layers} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL} TENSORBOARD=${TENSORBOARD} EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS} MILESTONE=${MILESTONE} EVAL_SEEDS=${EVAL_SEEDS} PYTHON_BIN=${PYTHON_BIN}" >> "${TASK_FILE}"
        done
      done
    done
  done
done

TASK_FILE="${TASK_FILE}" \
GPU_IDS="${GPU_IDS}" \
LOG_ROOT="${LOG_ROOT}" \
bash scripts/experiments/schedule_gpu_queue.sh
