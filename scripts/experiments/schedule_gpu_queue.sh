#!/usr/bin/env bash
set -euo pipefail

# Run a task file as a dynamic single-GPU queue.
# Each task line is a list of environment assignments, for example:
# MODE=forecast DATASET=weather HISTORY=96 HORIZON=192

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

TASK_FILE=${TASK_FILE:-}
TASK_RUNNER=${TASK_RUNNER:-"bash scripts/experiments/train_eval.sh"}
GPU_IDS=${GPU_IDS:-"0"}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
GPU_MAX_MEMORY_USED_MB=${GPU_MAX_MEMORY_USED_MB:-2048}
GPU_MAX_UTILIZATION=${GPU_MAX_UTILIZATION:-50}
REQUIRE_IDLE_GPU=${REQUIRE_IDLE_GPU:-true}
DRY_RUN=${DRY_RUN:-false}
LOG_ROOT=${LOG_ROOT:-logs/experiment_queue}
STATUS_CSV=${STATUS_CSV:-${LOG_ROOT}/status.csv}

if [[ -z "${TASK_FILE}" || ! -f "${TASK_FILE}" ]]; then
  echo "TASK_FILE is required and must point to a task list." >&2
  exit 1
fi

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

DRY_RUN_LOWER=$(lower "${DRY_RUN}")
REQUIRE_IDLE_GPU_LOWER=$(lower "${REQUIRE_IDLE_GPU}")

case "${DRY_RUN_LOWER}" in
  true|false) ;;
  *)
    echo "DRY_RUN must be true or false. Got ${DRY_RUN}" >&2
    exit 1
    ;;
esac

case "${REQUIRE_IDLE_GPU_LOWER}" in
  true|false) ;;
  *)
    echo "REQUIRE_IDLE_GPU must be true or false. Got ${REQUIRE_IDLE_GPU}" >&2
    exit 1
    ;;
esac

if [[ "${DRY_RUN_LOWER}" != "true" && "${REQUIRE_IDLE_GPU_LOWER}" == "true" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required when REQUIRE_IDLE_GPU=true." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
if [[ ! -f "${STATUS_CSV}" ]]; then
  echo "timestamp,event,task_id,gpu,pid,status,log_file,task" > "${STATUS_CSV}"
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_status() {
  local event=$1 task_id=$2 gpu=$3 pid=$4 status=$5 log_file=$6 task=$7
  echo "$(timestamp),${event},${task_id},${gpu},${pid},${status},${log_file},\"${task}\"" >> "${STATUS_CSV}"
}

task_value() {
  local task=$1 key=$2 part
  for part in ${task}; do
    case "${part}" in
      "${key}="*) echo "${part#*=}"; return 0 ;;
    esac
  done
  echo ""
}

task_id() {
  local index=$1 task=$2 mode model dataset seq history horizon emb_size num_layers model_size_tag
  mode=$(task_value "${task}" MODE)
  model=$(task_value "${task}" MODEL_NAME)
  dataset=$(task_value "${task}" DATASET)
  seq=$(task_value "${task}" SEQ_LENGTH)
  history=$(task_value "${task}" HISTORY)
  horizon=$(task_value "${task}" HORIZON)
  emb_size=$(task_value "${task}" EMB_SIZE)
  num_layers=$(task_value "${task}" NUM_LAYERS)
  model=${model:-model}
  model_size_tag=""
  if [[ -n "${emb_size}" && -n "${num_layers}" ]]; then
    model_size_tag="_emb${emb_size}_layers${num_layers}"
  fi
  case "${mode}" in
    uncond) echo "$(printf '%03d' "${index}")_${model}_${dataset}_seq${seq}${model_size_tag}" ;;
    cond) echo "$(printf '%03d' "${index}")_${model}_${dataset}_cond_seq${seq}" ;;
    forecast) echo "$(printf '%03d' "${index}")_${model}_${dataset}_forecast_h${history}_p${horizon}" ;;
    *) echo "$(printf '%03d' "${index}")_task" ;;
  esac
}

gpu_is_reserved() {
  local gpu=$1 index
  for index in "${!PIDS[@]}"; do
    if [[ "${PID_GPU[${index}]}" == "${gpu}" ]]; then
      return 0
    fi
  done
  return 1
}

gpu_is_idle() {
  local gpu=$1 line memory_used utilization
  if gpu_is_reserved "${gpu}"; then
    return 1
  fi
  if [[ "${REQUIRE_IDLE_GPU_LOWER}" != "true" ]]; then
    return 0
  fi

  line=$(nvidia-smi \
    --id="${gpu}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "${line}" ]]; then
    return 1
  fi

  memory_used=$(echo "${line}" | awk -F',' '{gsub(/ /, "", $1); print $1}')
  utilization=$(echo "${line}" | awk -F',' '{gsub(/ /, "", $2); print $2}')
  [[ "${memory_used}" -le "${GPU_MAX_MEMORY_USED_MB}" && "${utilization}" -le "${GPU_MAX_UTILIZATION}" ]]
}

launch_task() {
  local gpu=$1 index=$2 task=$3 id log_file
  local task_env runner
  id=$(task_id "${index}" "${task}")
  log_file="${LOG_ROOT}/${id}.log"

  read -r -a task_env <<< "${task}"
  read -r -a runner <<< "${TASK_RUNNER}"

  (
    echo "[$(timestamp)] START ${id} gpu=${gpu}"
    env CUDA_VISIBLE_DEVICES="${gpu}" GPU=0 "${task_env[@]}" "${runner[@]}"
    echo "[$(timestamp)] FINISH ${id} gpu=${gpu}"
  ) > "${log_file}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  PID_GPU+=("${gpu}")
  PID_ID+=("${id}")
  PID_TASK+=("${task}")
  PID_LOG+=("${log_file}")

  echo "START ${id}: gpu=${gpu} pid=${pid} log=${log_file}"
  log_status "start" "${id}" "${gpu}" "${pid}" "running" "${log_file}" "${task}"
}

reap_finished_jobs() {
  local index pid id gpu task log_file status
  for index in "${!PIDS[@]}"; do
    pid=${PIDS[${index}]}
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    id=${PID_ID[${index}]}
    gpu=${PID_GPU[${index}]}
    task=${PID_TASK[${index}]}
    log_file=${PID_LOG[${index}]}

    if wait "${pid}"; then
      status="success"
    else
      status="failed"
    fi

    echo "DONE ${id}: gpu=${gpu} pid=${pid} status=${status} log=${log_file}"
    log_status "finish" "${id}" "${gpu}" "${pid}" "${status}" "${log_file}" "${task}"

    unset "PIDS[${index}]"
    unset "PID_GPU[${index}]"
    unset "PID_ID[${index}]"
    unset "PID_TASK[${index}]"
    unset "PID_LOG[${index}]"
  done
}

TASKS=()
while IFS= read -r task_line; do
  [[ "${task_line}" =~ ^[[:space:]]*($|#) ]] && continue
  TASKS+=("${task_line}")
done < "${TASK_FILE}"
PIDS=()
PID_GPU=()
PID_ID=()
PID_TASK=()
PID_LOG=()

echo "GPU queue started at $(timestamp)"
echo "task_file=${TASK_FILE}"
echo "task_runner=${TASK_RUNNER}"
echo "gpu_ids=${GPU_IDS}"
echo "require_idle_gpu=${REQUIRE_IDLE_GPU_LOWER} memory_limit_mb=${GPU_MAX_MEMORY_USED_MB} utilization_limit_percent=${GPU_MAX_UTILIZATION} check_interval_seconds=${CHECK_INTERVAL}"
echo "queued_tasks=${#TASKS[@]}"
echo "status_csv=${STATUS_CSV}"

if [[ "${DRY_RUN_LOWER}" == "true" ]]; then
  read -r -a DRY_GPU_LIST <<< "${GPU_IDS}"
  if [[ "${#DRY_GPU_LIST[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU id." >&2
    exit 1
  fi

  for index in "${!TASKS[@]}"; do
    task=${TASKS[${index}]}
    gpu=${DRY_GPU_LIST[$((index % ${#DRY_GPU_LIST[@]}))]}
    id=$(task_id "${index}" "${task}")
    log_file="${LOG_ROOT}/${id}.log"
    echo "DRY_RUN ${id}: CUDA_VISIBLE_DEVICES=${gpu} GPU=0 ${task} ${TASK_RUNNER}"
    log_status "dry_run" "${id}" "${gpu}" "-" "planned" "${log_file}" "${task}"
  done

  echo "GPU queue finished at $(timestamp)"
  exit 0
fi

next_task=0
while [[ "${next_task}" -lt "${#TASKS[@]}" || "${#PIDS[@]}" -gt 0 ]]; do
  reap_finished_jobs

  while [[ "${next_task}" -lt "${#TASKS[@]}" ]]; do
    launched=false
    for gpu in ${GPU_IDS}; do
      if gpu_is_idle "${gpu}"; then
        launch_task "${gpu}" "${next_task}" "${TASKS[${next_task}]}"
        next_task=$((next_task + 1))
        launched=true
        break
      fi
    done

    if [[ "${launched}" != "true" ]]; then
      echo "WAIT: no available GPU; pending=$((${#TASKS[@]} - next_task)) running=${#PIDS[@]}. Recheck in ${CHECK_INTERVAL}s."
      if [[ "${REQUIRE_IDLE_GPU_LOWER}" == "true" ]]; then
        echo "GPU limits: memory <= ${GPU_MAX_MEMORY_USED_MB} MiB, utilization <= ${GPU_MAX_UTILIZATION}%."
        echo "Current GPUs (index, memory MiB, utilization %):"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
      fi
      break
    fi
  done

  if [[ "${next_task}" -lt "${#TASKS[@]}" || "${#PIDS[@]}" -gt 0 ]]; then
    sleep "${CHECK_INTERVAL}"
  fi
done

echo "GPU queue finished at $(timestamp)"
