#!/usr/bin/env bash
set -euo pipefail

# Queue conditional generation experiments across datasets and sequence lengths.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

DATASETS=${DATASETS:-"air electricity ettm1"}
SEQ_LENGTHS=${SEQ_LENGTHS:-"24 48 96"}
MODEL_NAME=${MODEL_NAME:-"quasid_cont quasid_disc"}
BACKBONE=${BACKBONE:-dit_solver_v5}
USE_PMI_CORR=${USE_PMI_CORR:-true}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_EVAL=${RUN_EVAL:-true}
TENSORBOARD=${TENSORBOARD:-true}
EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS:-true}
MILESTONE=${MILESTONE:-10}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_IDS=${GPU_IDS:-"0"}
RUN_ID=${RUN_ID:-"$(date '+%Y%m%d-%H%M%S')_pid$$"}
LOG_ROOT=${LOG_ROOT:-"logs/conditional_queue_v5/${RUN_ID}"}

TASK_FILE=$(mktemp "${TMPDIR:-/tmp}/quasid_conditional_tasks.XXXXXX")
trap 'rm -f "${TASK_FILE}"' EXIT

case "${EVAL_ALL_CHECKPOINTS}" in
  true|false) ;;
  *)
    echo "EVAL_ALL_CHECKPOINTS must be true or false. Got ${EVAL_ALL_CHECKPOINTS}" >&2
    exit 1
    ;;
esac

for dataset in ${DATASETS}; do
  config_file="config/conditional/${dataset}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "SKIP dataset=${dataset}: missing ${config_file}" >&2
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
    for seq_length in ${SEQ_LENGTHS}; do
      echo "MODE=cond DATASET=${dataset} SEQ_LENGTH=${seq_length} MODEL_NAME=${model_name} BACKBONE=${task_backbone} USE_PMI_CORR=${task_use_pmi_corr} RUN_TRAIN=${RUN_TRAIN} RUN_EVAL=${RUN_EVAL} TENSORBOARD=${TENSORBOARD} EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS} MILESTONE=${MILESTONE} PYTHON_BIN=${PYTHON_BIN}" >> "${TASK_FILE}"
    done
  done
done

TASK_FILE="${TASK_FILE}" \
GPU_IDS="${GPU_IDS}" \
LOG_ROOT="${LOG_ROOT}" \
bash scripts/experiments/schedule_gpu_queue.sh
