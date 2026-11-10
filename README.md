# QuaSiD

QuaSiD is a PyTorch implementation of frequency-domain time-series diffusion
with separate real and imaginary DiT streams and shared quadrature PMI correction.
This source release contains **the V5 backbone only**.

Supported tasks are unconditional generation, embedding-conditioned generation,
and probabilistic forecasting. The same V5 backbone is available with continuous
VP-SDE and discrete DDPM diffusion wrappers.

## Installation

Use Python 3.10 or 3.11. From the extracted `QuaSiD/` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your machine and CUDA driver. Runtime
dependencies and the test runner are listed together in `requirements.txt`.
No TensorFlow installation or separate FlashAttention package is needed.

CUDA configurations use `solver.mixed_precision: bf16` and PyTorch's flash SDPA
backend by default. For CPU checks or CUDA devices without the required flash
support, override `model.params.attn_backend sdpa` and
`solver.mixed_precision no`. Python's YAML loader treats the scalar `no` as false;
the trainer disables mixed precision in that case.

## Quick check without downloading data

```bash
python -m pytest -q
python scripts/smoke_v5.py
```

The smoke check uses temporary sine data on CPU, trains tiny V5 models with both
diffusion wrappers, saves checkpoints, and reloads them for sampling. It does not
run the expensive learned quality metrics or create artifacts inside this repository.

## Data

The repository includes 11 NPY datasets in `data/datasets/`: `Quito`, `air`,
`be`, `de`, `etth1`, `etth2`, `ettm1`, `ettm2`, `exchange`, `stocks`, and `weather`.
Synthetic sine data are generated in memory by the loader and need no dataset
download. Configurations for other datasets require you to provide their files.

Configurations are organized by task:

| Task | Mode | Configuration directory |
| --- | --- | --- |
| Unconditional generation | `uncond` | `config/unconditional/` |
| Embedding-conditioned generation | `cond` | `config/conditional/` |
| Probabilistic forecasting | `forecast` | `config/forecasting/` |

All bundled YAML files select `backbone: dit_solver_v5`. The Quito configuration
expects `data/datasets/Quito.npy`, shaped `[time, 5]`. Keep the capital `Q` in its
filename and configuration name.

Conditional embeddings, raw Parquet sources, checkpoints, and generated
samples are not bundled. Prepare any additional files required by your selected
configuration. The synthetic sine loader works without files.

### Unconditional generation

`TimeSeriesDataset` accepts numeric float arrays with either shape:

- `[time, channels]`: one continuous series;
- `[series, time, channels]`: independent series, each windowed separately.

Provide finite values and enough time steps for the configured window. Set
`dataloader.dataset.params.data_root`, `window`, `channels`, and `stride` in the
YAML. Match `model.params.seq_length` and `model.params.channels` to these values.
The loader fits normalization on the training windows. Bundled unconditional
configs use all windows (`split_ratio: 1.0`) and min-max scaling to `[-1, 1]`.

Example file names are `etth1.npy`, `stocks.npy`, `air.npy`, `eeg.npy`,
`energy.npy`, `fmri.npy`, and `Quito.npy`. The bundled Quito config expects five
channels; the source dataset used for that config has 15,356 rows. The Quito
Parquet extraction and raw Parquet source are not included.

`scripts/data/prepare_npy_datasets.py` converts supported CSV, MAT, and ARFF
inputs found in this directory. It also writes sine data and may replace
existing NPY outputs. The default sine configuration generates data directly
through `utils.sine_dataset.SineDataset`, independently of `sines.npy`.

### Embedding-conditioned generation

Place the configured CSV in `data/datasets/TSFragment-600K/` or change `data_root`.
The filename is `<name>.csv`, where `name` is the configured dataset name, such as
`embedding_cleaned_ETTh1_24`.

Required CSV columns:

| Column | Content |
| --- | --- |
| `OT` | A Python-list string of numeric target values, one sequence per row |
| `Text` | The text associated with that sequence |
| `TextEmbedding` | A serialized numeric embedding |

For the bundled benchmark names, `TextEmbedding` is parsed as whitespace-separated
numbers inside square brackets. Some domain-specific names use Python-list
syntax; see `ConditionalDataset.read_data` for the exact parser.

The loader returns targets `[length, 1]` and embeddings `[condition_dim]`.
Provide target lengths and embedding widths matching the YAML. The current
bundled embedding configurations use 128-dimensional embeddings and a seeded
90/10 train/test split. Embeddings must already exist; no text-embedding model
or embedding-generation service is bundled.

### Forecasting

The forecasting loader expects a numeric `[time, channels]` NPY array. Set
`history_length`, `pred_length`, `train_ratio`, and `val_ratio` in the data params.
The remainder of the time axis is used for the test split; normalization is fitted
on the training portion. See `ForecastingDataset` for window boundaries.

Set `model.params.seq_length = history_length + pred_length`. The launchers
derive this automatically when `HISTORY` and `HORIZON` are selected. Frequency
condition widths are derived at runtime and should not be added to forecast YAML.

## Train and evaluate

The following commands run from the repository root. Unconditional sine generation
is a complete example with no external data:

```bash
python main.py --train --name sines_v5 --mode uncond \
  --config_file config/unconditional/sines.yaml --model_name pacodi_sde

python main.py --name sines_v5 --mode uncond \
  --config_file config/unconditional/sines.yaml --model_name pacodi_sde \
  --milestone 10 --seed 2021
```

Use an existing checkpoint milestone. Milestones increase at every
`solver.save_cycle`; for example, 20,000 steps with a 2,000-step save cycle
produces checkpoints 1 through 10. Evaluation loads the EMA model and computes
metrics and visualizations unless `--skip_metrics` or `--skip_visualization` is set.
It generates samples in memory; the evaluation entry points do not export a
standalone generated-sample NPY file.

Options after the standard flags are YAML overrides in `key value` pairs:

```bash
python main.py --train --name Quito_v5 --mode uncond \
  --config_file config/unconditional/Quito.yaml --model_name pacodi_sde \
  model.params.seq_length 128 dataloader.dataset.params.window 128

# Conditional data and forecasting data must first be provided as described above.
MODE=cond DATASET=etth1 RUN_EVAL=false \
  bash scripts/experiments/train_eval.sh

MODE=forecast DATASET=etth1 RUN_EVAL=false \
  bash scripts/experiments/train_eval.sh
```

To select the DDPM wrapper, use `MODEL_NAME=pacodi_ddpm` with the shell launchers.
With `main.py`, set both `--model_name pacodi_ddpm` and the YAML override
`model.target models.pacodi.pacodi_ddpm.PaCoDi_ddpm`.

The internal module directory `models/pacodi/`, class names, and the identifiers
`pacodi_sde` / `pacodi_ddpm` are retained for checkpoint compatibility. They refer
to the two diffusion wrappers around the sole V5 backbone in this QuaSiD release.
Existing V5 checkpoints require their original architecture and data settings.

## GPU queues and repeated evaluation

```bash
# Preview the V5 Quito tasks. The preview writes only queue bookkeeping logs.
DRY_RUN=true DATASETS=Quito GPU_IDS=0 \
  bash scripts/experiments/run_unconditional.sh

# Train one V5 configuration and evaluate every saved checkpoint.
DATASETS=Quito SEQ_LENGTHS=128 MODEL_NAME=pacodi_sde GPU_IDS=0 \
  bash scripts/experiments/run_unconditional.sh

# Forecasting and conditional queues.
bash scripts/experiments/run_conditional.sh
bash scripts/experiments/run_forecasting.sh
```

Set `PYTHON_BIN` to the Python interpreter in your environment. Queue launchers
default to GPU `0`; use `GPU_IDS="0 1"` for two workers. They wait for idle GPUs
according to `GPU_MAX_MEMORY_USED_MB` and `GPU_MAX_UTILIZATION`.
`REQUIRE_IDLE_GPU=false` disables that idle check.

Useful settings include `DATASETS`, `SEQ_LENGTHS`, `MODEL_NAME`, `RUN_TRAIN`,
`RUN_EVAL`, `EVAL_ALL_CHECKPOINTS`, `MILESTONE`, `EVAL_SEEDS`, and `LOG_ROOT`.
The supported model names are `pacodi_sde` and `pacodi_ddpm`;
`BACKBONE` must remain `dit_solver_v5`.

For a saved unconditional run, reproduce its original training overrides and use
`RUN_TRAIN=false` with the same `EXP_NAME` in `train_eval.sh`.
`EVAL_SEEDS` is a comma-separated list. The summary script reports statistics
across these evaluation seeds, not across independently trained models.
Learned metrics such as Context-FID, discriminative score, and predictive score
train additional networks and may take substantial time.

## Repository layout

```text
main.py                    Training and checkpoint-evaluation CLI
models/pacodi/             V5 backbone, shared blocks, SDE/DDPM wrappers
models/ts2vec/             Context-FID representation model
engine/                    Optimizer loop, EMA integration, schedulers, logging
data/                      Loaders and normalization
config/                    V5 configurations for all three tasks
evaluation/                Task evaluation and metrics output
utils/                     Metrics, EMA, visualization, runtime utilities
scripts/data/              Optional data conversion and sine generation tools
scripts/experiments/       GPU queues, train/evaluate launchers, metric summaries
tests/                     V5, EMA, sine-loader, and metric-space checks
```

Outputs go to `experiments/<mode>/<model_name>/<dataset>/<experiment>/`, including
checkpoints, logs, evaluation plots, and metrics CSVs. Queue logs go to `logs/`.
These generated directories are ignored by Git.

## Source release and validation

This repository contains the V5 backbone and its required training, data, and
evaluation dependencies. Earlier backbones and comparison implementations are
excluded, along with checkpoints, logs, and private remote-operation scripts.
The 11 NPY datasets listed above are included. All bundled model configurations
select V5.

The initial source release passed 41 tests, CPU training/checkpoint/reload/sampling
checks for both diffusion wrappers, and schema/import checks for all 25 YAML
configurations. Eight fixed-seed wrapper/task comparisons matched the original
V5 source exactly in loss, model state, and sampled outputs. These checks used
Python 3.10.20, PyTorch 2.5.1, and NumPy 1.26.4; they did not include full
benchmark retraining or a fresh dependency installation. Run the commands in
the quick-check section to verify your own environment.
