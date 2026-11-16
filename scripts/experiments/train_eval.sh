#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

MODE=${MODE:-uncond}
DATASET=${DATASET:-}
MODEL_NAME=${MODEL_NAME:-}
MILESTONE=${MILESTONE:-}
if [[ -z "${MILESTONE}" ]]; then
  if [[ "${MODE}" == "forecast" ]]; then MILESTONE=best; else MILESTONE=5; fi
fi
EVAL_ALL_CHECKPOINTS=${EVAL_ALL_CHECKPOINTS:-false}
GPU=${GPU:-0}
RUN_TRAIN=${RUN_TRAIN:-true}
RUN_EVAL=${RUN_EVAL:-true}
TENSORBOARD=${TENSORBOARD:-true}
EVAL_SEEDS=${EVAL_SEEDS:-12345}
SUMMARIZE_EVALS=${SUMMARIZE_EVALS:-true}
PYTHON_BIN=${PYTHON_BIN:-python}
export QUASID_TENSORBOARD_ROOT=${QUASID_TENSORBOARD_ROOT:-/tmp/quasid-tensorboard}

case "${MODE}" in
  uncond|cond|forecast) ;;
  *)
    echo "MODE must be one of: uncond, cond, forecast. Got ${MODE}" >&2
    exit 1
    ;;
esac

if [[ -z "${MODEL_NAME}" ]]; then
  case "${MODE}" in
    uncond|cond) MODEL_NAME=quasid_cont ;;
    forecast) MODEL_NAME=quasid_cont ;;
  esac
fi

case "${MODEL_NAME}" in
  quasid_cont)
    MODEL_TARGET=models.quasid.quasid_cont.QuaSiD_cont
    ;;
  quasid_disc)
    MODEL_TARGET=models.quasid.quasid_disc.QuaSiD_disc
    ;;
  *)
    echo "MODEL_NAME must be quasid_cont or quasid_disc. Got ${MODEL_NAME}" >&2
    exit 1
    ;;
esac

if [[ -z "${DATASET}" ]]; then
  echo "DATASET is required, e.g. DATASET=stocks MODE=uncond." >&2
  exit 1
fi

config_path() {
  case "${MODE}" in
    uncond) echo "config/unconditional/${DATASET}.yaml" ;;
    cond) echo "config/conditional/${DATASET}.yaml" ;;
    forecast) echo "config/forecasting/${DATASET}.yaml" ;;
  esac
}

config_file=$(config_path)

if [[ ! -f "${config_file}" ]]; then
  echo "Config not found for MODE=${MODE}, DATASET=${DATASET}: ${config_file}" >&2
  exit 1
fi

yaml_scalar() {
  local key=$1
  awk -v key="${key}:" '$1 == key {print $2; exit}' "${config_file}"
}

case "${RUN_TRAIN}" in
  true|false) ;;
  *)
    echo "RUN_TRAIN must be true or false. Got ${RUN_TRAIN}" >&2
    exit 1
    ;;
esac

case "${RUN_EVAL}" in
  true|false) ;;
  *)
    echo "RUN_EVAL must be true or false. Got ${RUN_EVAL}" >&2
    exit 1
    ;;
esac

case "${TENSORBOARD}" in
  true|false) ;;
  *)
    echo "TENSORBOARD must be true or false. Got ${TENSORBOARD}" >&2
    exit 1
    ;;
esac

case "${EVAL_ALL_CHECKPOINTS}" in
  true|false) ;;
  *)
    echo "EVAL_ALL_CHECKPOINTS must be true or false. Got ${EVAL_ALL_CHECKPOINTS}" >&2
    exit 1
    ;;
esac

case "${SUMMARIZE_EVALS}" in
  true|false) ;;
  *)
    echo "SUMMARIZE_EVALS must be true or false. Got ${SUMMARIZE_EVALS}" >&2
    exit 1
    ;;
esac

# The queue serializes task environment variables by whitespace, so this value
# is deliberately comma-separated rather than space-separated.
IFS=',' read -r -a raw_eval_seeds <<< "${EVAL_SEEDS}"
EVAL_SEED_LIST=()
declare -A seen_eval_seeds=()
for raw_seed in "${raw_eval_seeds[@]}"; do
  eval_seed=${raw_seed//[[:space:]]/}
  if [[ ! "${eval_seed}" =~ ^[0-9]+$ ]] || [[ ${#eval_seed} -gt 10 ]] || (( 10#${eval_seed} > 4294967295 )); then
    echo "EVAL_SEEDS must be comma-separated integers in [0, 4294967295]. Got ${EVAL_SEEDS}" >&2
    exit 1
  fi
  if [[ -n "${seen_eval_seeds[${eval_seed}]+x}" ]]; then
    echo "EVAL_SEEDS must not contain duplicates. Got ${EVAL_SEEDS}" >&2
    exit 1
  fi
  seen_eval_seeds[${eval_seed}]=1
  EVAL_SEED_LIST+=("${eval_seed}")
done

if [[ ${#EVAL_SEED_LIST[@]} -eq 0 ]]; then
  echo "EVAL_SEEDS must contain at least one integer." >&2
  exit 1
fi

SEQ_LENGTH=${SEQ_LENGTH:-$(yaml_scalar seq_length)}
BACKBONE=${BACKBONE:-$(yaml_scalar backbone)}
USE_PMI_CORR=${USE_PMI_CORR:-$(yaml_scalar use_pmi_corr)}
HISTORY=${HISTORY:-$(yaml_scalar history_length)}
HORIZON=${HORIZON:-$(yaml_scalar pred_length)}
CONDITION_DOMAIN=${CONDITION_DOMAIN:-$(yaml_scalar condition_domain)}
CHANNEL_INDEP=${CHANNEL_INDEP:-$(yaml_scalar channel_indep)}
CHANNEL_INDEP=${CHANNEL_INDEP:-false}
MODEL_CHANNELS=$(yaml_scalar channels)
EMB_SIZE=${EMB_SIZE:-$(yaml_scalar emb_size)}
NUM_LAYERS=${NUM_LAYERS:-$(yaml_scalar num_layers)}
CONDITIONAL_DATASET_NAME=${CONDITIONAL_DATASET_NAME:-}


if [[ "${MODE}" == "forecast" ]]; then
  if [[ -z "${HISTORY}" || -z "${HORIZON}" ]]; then
    echo "Could not infer HISTORY or HORIZON from ${config_file}" >&2
    exit 1
  fi

  SEQ_LENGTH=$((HISTORY + HORIZON))
  CONDITION_DOMAIN=${CONDITION_DOMAIN:-frequency}
  case "${CONDITION_DOMAIN}" in
    frequency|time) ;;
    *)
      echo "condition_domain must be frequency or time. Got ${CONDITION_DOMAIN}" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${SEQ_LENGTH}" || -z "${BACKBONE}" || -z "${USE_PMI_CORR}" ]]; then
  echo "Could not infer SEQ_LENGTH, BACKBONE, or USE_PMI_CORR from ${config_file}" >&2
  exit 1
fi

if [[ "${MODE}" == "cond" && -z "${CONDITIONAL_DATASET_NAME}" ]]; then
  config_dataset_name=$(yaml_scalar name)
  if [[ "${config_dataset_name}" =~ ^(.+)_([0-9]+)$ ]]; then
    CONDITIONAL_DATASET_NAME="${BASH_REMATCH[1]}_${SEQ_LENGTH}"
  else
    # Some custom conditional datasets use one file for every sequence length.
    # Keep their configured name unchanged unless an explicit override is given.
    CONDITIONAL_DATASET_NAME="${config_dataset_name}"
  fi
fi

if [[ ! "${EMB_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EMB_SIZE must be a positive integer. Got ${EMB_SIZE}" >&2
  exit 1
fi

if [[ ! "${NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_LAYERS must be a positive integer. Got ${NUM_LAYERS}" >&2
  exit 1
fi

if [[ "${BACKBONE}" == dit_solver_v* ]]; then
  DEFAULT_BACKBONE_TAG="ditv${BACKBONE#dit_solver_v}"
else
  DEFAULT_BACKBONE_TAG="${BACKBONE}"
fi
: "${BACKBONE_TAG:=${DEFAULT_BACKBONE_TAG}}"

USE_PMI_CORR_LOWER=$(printf '%s' "${USE_PMI_CORR}" | tr '[:upper:]' '[:lower:]')
case "${USE_PMI_CORR_LOWER}" in
  true|1|yes|y|on)
    USE_PMI_CORR=true
    CORR_TAG=it1
    ;;
  false|0|no|n|off)
    USE_PMI_CORR=false
    CORR_TAG=it0
    ;;
  *)
    echo "USE_PMI_CORR must be true or false. Got ${USE_PMI_CORR}" >&2
    exit 1
    ;;
esac

auto_experiment_name() {
  case "${MODE}" in
    uncond)
      echo "${DATASET}_seq${SEQ_LENGTH}_${BACKBONE_TAG}_${CORR_TAG}_emb${EMB_SIZE}_layers${NUM_LAYERS}"
      ;;
    cond)
      echo "${DATASET}_cond_seq${SEQ_LENGTH}_${BACKBONE_TAG}_${CORR_TAG}"
      ;;
    forecast)
      echo "${DATASET}_forecast_h${HISTORY}_p${HORIZON}_${BACKBONE_TAG}_${CORR_TAG}"
      ;;
  esac
}

EXP_NAME=${EXP_NAME:-$(auto_experiment_name)}

PYTHON_CMD=()
PYTHON_CMD=("${PYTHON_BIN}")
# Keep CLI flags before argparse's remainder; config overrides remain last.
CLI_EXTRA_ARGS=()
EXTRA_OPTS=()
while (( $# )); do
  case "$1" in
    --) shift; EXTRA_OPTS+=("$@"); break ;;
    --skip_metrics|--skip_visualization)
      CLI_EXTRA_ARGS+=("$1"); shift ;;
    --*)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      CLI_EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    *) EXTRA_OPTS+=("$1"); shift ;;
  esac
done

GPU_ARGS=()
if [[ -n "${GPU}" ]]; then
  GPU_ARGS=(--gpu "${GPU}")
fi

TENSORBOARD_ARGS=()
if [[ "${TENSORBOARD}" == "true" ]]; then
  TENSORBOARD_ARGS=(--tensorboard)
fi

COMMON_ARGS=(
  --mode "${MODE}"
  --config_file "${config_file}"
  --model_name "${MODEL_NAME}"
  --dataset_name "${DATASET}"
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"}
  ${TENSORBOARD_ARGS[@]+"${TENSORBOARD_ARGS[@]}"}
)

AUTO_OPTS=(
  model.target "${MODEL_TARGET}"
  model.params.seq_length "${SEQ_LENGTH}"
  model.params.backbone "${BACKBONE}"
  model.params.use_pmi_corr "${USE_PMI_CORR}"
  model.params.emb_size "${EMB_SIZE}"
  model.params.num_layers "${NUM_LAYERS}"
)

case "${MODE}" in
  uncond)
    AUTO_OPTS+=(
      dataloader.dataset.params.window "${SEQ_LENGTH}"
    )
    ;;
  cond)
    AUTO_OPTS+=(
      dataloader.dataset.params.window "${SEQ_LENGTH}"
      dataloader.dataset.params.name "${CONDITIONAL_DATASET_NAME}"
    )
    ;;
  forecast)
    AUTO_OPTS+=(
      model.params.channels "${MODEL_CHANNELS}"
      model.params.condition_domain "${CONDITION_DOMAIN}"
      model.params.channel_indep "${CHANNEL_INDEP}"
      dataloader.dataset.params.history_length "${HISTORY}"
      dataloader.dataset.params.pred_length "${HORIZON}"
    )
    ;;
esac

if [[ "${RUN_TRAIN}" == "true" ]]; then
  echo ">>> Train model=${MODEL_NAME} experiment=${EXP_NAME}"
  "${PYTHON_CMD[@]}" main.py \
    --train \
    --name "${EXP_NAME}" \
    "${COMMON_ARGS[@]}" \
    ${CLI_EXTRA_ARGS[@]+"${CLI_EXTRA_ARGS[@]}"} \
    "${AUTO_OPTS[@]}" \
    ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"}
fi

evaluate_checkpoint() {
  local milestone=$1
  local eval_seed=$2
  echo ">>> Eval model=${MODEL_NAME} experiment=${EXP_NAME} checkpoint=${EXP_NAME} milestone=${milestone} seed=${eval_seed}"
  "${PYTHON_CMD[@]}" main.py \
    --name "${EXP_NAME}" \
    --ckpt_name "${EXP_NAME}" \
    --milestone "${milestone}" \
    --seed "${eval_seed}" \
    --eval_tag "milestone-${milestone}-seed-${eval_seed}" \
    "${COMMON_ARGS[@]}" \
    ${CLI_EXTRA_ARGS[@]+"${CLI_EXTRA_ARGS[@]}"} \
    "${AUTO_OPTS[@]}" \
    ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"}
}

evaluate_checkpoint_seeds() {
  local milestone=$1
  local eval_seed
  for eval_seed in "${EVAL_SEED_LIST[@]}"; do
    evaluate_checkpoint "${milestone}" "${eval_seed}"
  done
}

summarize_checkpoint_evaluations() {
  local milestone=$1
  local metrics_path summary_path

  if [[ "${SUMMARIZE_EVALS}" != "true" ]]; then
    return
  fi
  if [[ ${#EVAL_SEED_LIST[@]} -lt 2 ]]; then
    echo "Skipping metric summary: at least two EVAL_SEEDS are required for sample variance."
    return
  fi

  metrics_path="experiments/${MODE}/${MODEL_NAME}/${DATASET}/${EXP_NAME}/metrics.csv"
  summary_path="experiments/${MODE}/${MODEL_NAME}/${DATASET}/${EXP_NAME}/metrics_summary_milestone-${milestone}.csv"
  "${PYTHON_CMD[@]}" scripts/experiments/summarize_metrics.py \
    --metrics-path "${metrics_path}" \
    --output "${summary_path}" \
    --experiment "${EXP_NAME}" \
    --milestone "${milestone}" \
    --seeds "${EVAL_SEED_LIST[@]}"
}

if [[ "${RUN_EVAL}" == "true" ]]; then
  if [[ "${EVAL_ALL_CHECKPOINTS}" == "true" ]]; then
    checkpoint_dir="experiments/${MODE}/${MODEL_NAME}/${DATASET}/${EXP_NAME}/ckpt"
    milestones=()
    for checkpoint_path in "${checkpoint_dir}"/ckpt-*.pt; do
      [[ -f "${checkpoint_path}" ]] || continue
      checkpoint_file=${checkpoint_path##*/}
      if [[ "${checkpoint_file}" =~ ^ckpt-([0-9]+)\.pt$ ]]; then
        milestones+=("${BASH_REMATCH[1]}")
      fi
    done

    if [[ ${#milestones[@]} -eq 0 ]]; then
      echo "No checkpoints found in ${checkpoint_dir}" >&2
      exit 1
    fi

    while IFS= read -r milestone; do
      evaluate_checkpoint_seeds "${milestone}"
      summarize_checkpoint_evaluations "${milestone}"
    done < <(printf '%s\n' "${milestones[@]}" | sort -n)
  else
    evaluate_checkpoint_seeds "${MILESTONE}"
    summarize_checkpoint_evaluations "${MILESTONE}"
  fi
fi
