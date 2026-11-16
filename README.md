# QuaSiD

QuaSiD is a PyTorch implementation of frequency-domain time-series diffusion
with separate real and imaginary DiT streams and shared quadrature PMI correction.

Supported tasks are unconditional generation, embedding-conditioned generation, and probabilistic forecasting. The same QuaSiD backbone is available with continuous VP-SDE and discrete DDPM diffusion wrappers.

## Model implementation

All model code lives in [models/quasid/](models/quasid/). The public model identifiers and configuration targets are:

| Diffusion wrapper | Model identifier | YAML `model.target` |
| --- | --- | --- |
| Continuous VP-SDE | `quasid_cont` | `models.quasid.quasid_cont.QuaSiD_cont` |
| Discrete DDPM | `quasid_disc` | `models.quasid.quasid_disc.QuaSiD_disc` |

[quasid.py](models/quasid/quasid.py) provides the shared `QuaSiD` base class, condition encoder, orthonormal rFFT transforms, spectral noise covariance, and noise-prediction loss. The [QuaSiD backbone](models/quasid/dit_solver_v5.py) implements the separate real and imaginary DiT streams; [dit_blocks.py](models/quasid/dit_blocks.py) contains their shared quadrature PMI correction. Both diffusion wrappers select their generation integrator through `model.params.sampler`.

### Blockwise PMI correction

Each DiT layer contains one `SharedQuadraturePMIBlock` operating on that layer's even/odd hidden states. The block uses two shared frequency encoders, six time/condition modulation vectors, and two linear decoders. Each branch fuses the correction as `hidden = hidden_dit - noise_scale * variance * score_pmi` before passing its output to the next layer. PMI modulation starts at zero, and the imaginary DC/Nyquist mask enforces the spectral constraints. Set `model.params.use_pmi_corr=false` to disable PMI.

### Sampling configuration

Set the sampling options under the existing `model.params` mapping for either diffusion wrapper:

```yaml
model:
  params:
    sampler: ode          # ode or sde
    sampling_steps: 200
```

`model.target` selects the continuous VP-SDE or discrete DDPM training schedule. `sampler` selects how the trained model generates samples:

| `sampler` | `quasid_cont` | `quasid_disc` |
| --- | --- | --- |
| `ode` | Probability-flow ODE integrated by continuous DDIM updates in `_ode_step()` | Deterministic DDIM with `eta=0` |
| `sde` | Reverse SDE integrated by Euler–Maruyama updates in `_euler_maruyama_step()` | Stochastic generalized DDIM jumps with `eta=1` |

Both wrappers dispatch from `_sample_spectrum()`. The [continuous wrapper](models/quasid/quasid_cont.py) selects `_ode_step()` or `_euler_maruyama_step()`. The [discrete wrapper](models/quasid/quasid_disc.py) calls `_ddpm_step()` and `DDPM.ddim_sample()`, with `eta` determined by `sampler`.

The `ode` options start from a random Gaussian state and add no fresh noise during integration. The `sde` options inject fresh noise at intermediate updates. The continuous SDE sampler always keeps only the drift update on its final step. Continuous ODE sampling and both discrete modes finish with a clean-signal prediction. Changing the seed can therefore change generated samples in either mode.

- `sampling_steps` controls the number of reverse-generation updates. The discrete wrapper limits this value to the range from 1 to `max_step`.
- `max_step` sets the discrete diffusion-chain length for `quasid_disc` and the backbone time-embedding scale for `quasid_cont`. `solver.train_steps` sets the number of optimizer updates.

Bundled YAML files declare `sampler` explicitly. If it is omitted, `quasid_cont` defaults to `sde`, and `quasid_disc` defaults to `ode`. Switch `sampler` or `sampling_steps` at evaluation time using a checkpoint trained with the same diffusion wrapper, architecture, and noise schedule.

## Installation

Use Python 3.10 or 3.11. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with your machine and CUDA driver. Runtime
dependencies and the test runner are listed together in `requirements.txt`.
No TensorFlow installation or separate FlashAttention package is needed.

The experiment shell launchers require Bash 4 or newer because they use associative arrays. Use a compatible Bash on macOS, where the system Bash may be older. GPU queues require NVIDIA GPUs and `nvidia-smi`; the CPU smoke check runs directly through Python.

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

The smoke check uses temporary sine data on CPU, trains tiny QuaSiD models with both diffusion wrappers, saves checkpoints, and reloads them for sampling. It does not run the expensive learned quality metrics or create artifacts inside this repository.

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

All bundled YAML files use the QuaSiD backbone. The Quito configuration expects `data/datasets/Quito.npy`, shaped `[time, 5]`. Keep the capital `Q` in its filename and configuration name.

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

The forecasting loader accepts numeric NPY arrays shaped `[time, channels]` or
`[series, time, channels]`. Set `history_length` and `pred_length`, then choose
`train_ratio` / `val_ratio` or `split_lengths: [train, val, test]`. Explicit
lengths take precedence and exclude any trailing data beyond their sum.
ETTh1/ETTh2 use `[8640, 2880, 2880]`; ETTm1/ETTm2 use
`[34560, 11520, 11520]`. Other bundled forecast configs use a 70/10/20 time split.
Normalization is fitted only on the training time segment. Validation and test
windows retain preceding history while their entire predicted future stays
inside the selected split. See [ForecastingDataset](data/dataset_utils/forecasting_dataset.py).

Forecast training builds a validation loader with ordered, complete batches.
`solver.validation.enabled`, `interval`, and `max_batches` control validation;
`max_batches: null` evaluates the full validation set. The bundled configs train
for 20,000 optimizer steps, save and validate every 2,000 steps, and use training
batch size 32 and validation/test batch size 64. A final validation runs if the
last step was not already validated.

Validation evaluates the **current model weights**, with sample-weighted mean
model loss. Classifier-free condition dropout applies only during training;
validation retains the supplied history condition. It does not generate
forecast ensembles or select by CRPS/MAE.
A strictly lower finite validation loss replaces `ckpt-best.pt`;
`ckpt-best.json` records the loss and selected step. The PT format remains
exactly `step` and `model`, and evaluation loads those selected model weights.
Forecast configs enable `solver.validation.early_stopping` with `enabled: true`
and `patience: 2`. After establishing a best loss, two consecutive validations
without a strictly lower loss stop training; ties and non-finite losses count as
non-improvements. A new best resets the counter. For example, with validation
every 2,000 steps, a best at step 2,000 followed by no improvement at steps 4,000
and 6,000 stops at step 6,000. Evaluation still loads the best checkpoint.
Set `enabled: false` to run the full step budget. Early stopping is inactive
without a validation loader. The learning-rate scheduler still receives training loss.

Forecast training writes `ckpt/training-complete.json` only after reaching the
step budget and final validation, or after validation-based early stopping. Early
stop records include the requested budget, patience, and validation settings.
The launcher reuses an existing checkpoint when this record reaches the effective
`train_steps`, or records early stopping under the same budget and stopping
settings, including CLI overrides. An early `ckpt-best.pt` alone never marks a run complete.
Incomplete runs and older runs without a completion record keep their checkpoints
and stop with a message. Use a new `EXP_NAME` for fresh training or
`RUN_TRAIN=false` to evaluate existing checkpoints; optimizer-state resume is
not supported by the current checkpoint format.

Forecast launchers default to `MILESTONE=best`. The queue defaults to history 168
and horizon 192; `national_illness` (and its `ili` alias) uses horizon 36, with
separate configured NPY paths. It generates 20 paths per condition in chunks of
4 and evaluates at most 20,000 uniformly selected test windows. Override these
with `HISTORY_LENGTHS`, `PRED_LENGTHS`, `NATIONAL_ILLNESS_PRED_LENGTHS`,
`FORECAST_SAMPLES`, `ENSEMBLE_CHUNK_SIZE`, and `FORECAST_EVAL_SAMPLE_LIMIT`.
Each generated target contains history plus future; metrics use only the future
in dataset-normalized space. The queue's length defaults override YAML lengths;
`train_eval.sh` used directly retains YAML lengths unless `HISTORY`/`HORIZON` are set.

```bash
# Train and test the validation-best forecasting checkpoint.
MODE=forecast DATASET=etth1 bash scripts/experiments/train_eval.sh

# Evaluate an existing best checkpoint across generation seeds.
MODE=forecast DATASET=etth1 RUN_TRAIN=false MILESTONE=best \
  EVAL_SEEDS=2022,2023,2024 bash scripts/experiments/train_eval.sh
```

Set `model.params.seq_length = history_length + pred_length`. The launchers
derive this automatically when `HISTORY` and `HORIZON` are selected. Frequency
condition widths are derived at runtime and should not be added to forecast YAML.

## Train and evaluate

The following commands run from the repository root. Unconditional sine generation
is a complete example with no external data:

```bash
python main.py --train --name sines_quasid --mode uncond \
  --config_file config/unconditional/sines.yaml --model_name quasid_cont

python main.py --name sines_quasid --mode uncond \
  --config_file config/unconditional/sines.yaml --model_name quasid_cont \
  --milestone 10 --seed 2021
```

Use an existing checkpoint milestone. Milestones increase at every
`solver.save_cycle`; for example, 20,000 steps with a 2,000-step save cycle
produces checkpoints 1 through 10. Evaluation loads the saved model and computes
metrics and visualizations unless `--skip_metrics` or `--skip_visualization` is set.
It generates samples in memory; the evaluation entry points do not export a
standalone generated-sample NPY file.

Options after the standard flags are YAML overrides in `key value` pairs:

```bash
python main.py --train --name Quito_quasid --mode uncond \
  --config_file config/unconditional/Quito.yaml --model_name quasid_cont \
  model.params.seq_length 128 dataloader.dataset.params.window 128

# Conditional generation requires the embedding CSV described above.
MODE=cond DATASET=etth1 RUN_EVAL=false \
  bash scripts/experiments/train_eval.sh

# Forecasting uses the bundled data/datasets/etth1.npy.
MODE=forecast DATASET=etth1 RUN_EVAL=false \
  bash scripts/experiments/train_eval.sh
```

### Select the diffusion wrapper

The shell launchers set both the model identifier and configuration target from `MODEL_NAME`. For example:

```bash
MODE=uncond DATASET=sines MODEL_NAME=quasid_disc RUN_EVAL=false \
  bash scripts/experiments/train_eval.sh
```

In `main.py`, `--model_name` controls output directory naming, while `model.target` selects the model class. Set both when switching from the bundled SDE configuration to DDPM, and keep YAML overrides after all CLI flags:

```bash
python main.py --train --name sines_quasid --mode uncond \
  --config_file config/unconditional/sines.yaml --model_name quasid_disc \
  model.target models.quasid.quasid_disc.QuaSiD_disc
```

Use the same target override when evaluating that DDPM checkpoint.

### Loading checkpoints

To load a QuaSiD checkpoint, match its original architecture, diffusion wrapper, and data settings. Use the corresponding `model.target` above and place the checkpoint at:

```text
experiments/<mode>/<model_name>/<dataset>/<experiment>/ckpt/ckpt-<milestone>.pt
```

For evaluation, `--ckpt_name` selects the checkpoint experiment and defaults to `--name`.

Main-model checkpoints contain only `step` and `model`. Evaluation uses the saved model parameters. Fresh training starts with a new optimizer; optimizer state, EMA state, and training-resume metadata are excluded from checkpoint storage.

## GPU queues and repeated evaluation

```bash
# Preview the Quito tasks. The preview writes only queue bookkeeping logs.
DRY_RUN=true DATASETS=Quito GPU_IDS=0 \
  bash scripts/experiments/run_unconditional.sh

# Train one QuaSiD configuration and evaluate every saved checkpoint.
DATASETS=Quito SEQ_LENGTHS=128 MODEL_NAME=quasid_cont GPU_IDS=0 \
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
The supported model names are `quasid_cont` and `quasid_disc`. Keep `BACKBONE` at its bundled default.

For a saved unconditional run, reproduce its original training overrides and use
`RUN_TRAIN=false` with the same `EXP_NAME` in `train_eval.sh`.
`EVAL_SEEDS` is a comma-separated list. The summary script reports statistics
across these evaluation seeds, not across independently trained models.
Context-FID, discriminative score (DS), and predictive score (PS) share evaluator checkpoints across generation seeds and models for the same dataset and horizon. Weights are loaded from `experiments/metric_checkpoints/`; a missing evaluator is trained once and saved.

Checkpoints use the fixed paths `experiments/metric_checkpoints/<dataset>/horizon_<length>/ds.pt`, `ps.pt`, and `ts2vec.pt`. Dataset names match the experiment's `dataset_name`, including case. For example, Quito at horizon 512 uses `experiments/metric_checkpoints/Quito/horizon_512/ds.pt`. Existing weights are reused across generation seeds and models.

The dataset and horizon select the evaluator checkpoint. Training settings apply when the checkpoint is missing. The first generated batch supplies the training data for DS and PS; TS2Vec trains on the real reference only.

With a fixed predictor and real reference, PS reports the same real-data MAE for each generated batch. DS and Context-FID evaluate each new generated batch.

## Repository layout

```text
main.py                    Training and checkpoint-evaluation CLI
models/quasid/             QuaSiD base class, DiT backbone, SDE/DDPM wrappers
models/ts2vec/             Context-FID representation model
engine/                    Optimizer loop, EMA integration, schedulers, logging
data/                      Loaders and normalization
config/                    QuaSiD configurations for all three tasks
evaluation/                Task evaluation and metrics output
utils/                     Metrics, EMA, visualization, runtime utilities
scripts/data/              Optional data conversion and sine generation tools
scripts/experiments/       GPU queues, train/evaluate launchers, metric summaries
tests/                     Model, EMA, sine-loader, and metric-space checks
```

Outputs go to `experiments/<mode>/<model_name>/<dataset>/<experiment>/`, including
checkpoints, logs, evaluation plots, and metrics CSVs. Queue logs go to `logs/`.
These generated directories are ignored by Git.

## Development workflow

For SSH setup, reuse in future projects, and authentication troubleshooting, see the [AntCode Git SSH guide](aistudio/GIT_SSH_SETUP.md). The [Git and server synchronization workflow](aistudio/GIT_PUSH_AND_SERVER_SYNC_SOP.md) describes local checks, commits, pushes, and separately authorized server uploads.

## Source release and validation

This repository contains the QuaSiD model, training and evaluation code, task configurations, and the 11 NPY datasets listed above. Checkpoints, logs, and local authentication credentials are excluded.

Historical validation of the blockwise-only release included:

- All 41 tests covering the QuaSiD backbone, diffusion wrappers, EMA, sine loader, and conditional metric scaling.
- CPU training, finite checkpoint checks, model reload, and sampling for both diffusion wrappers.
- Schema validation and model-target imports for all 25 YAML configurations.
- Shell syntax checks for the experiment launchers.

These historical checks ran in the existing `rycai` conda environment with Python 3.11.15, PyTorch 2.12.1, and NumPy 2.4.6. That environment's NumPy version is outside the installation range in `requirements.txt`. They do not validate a fresh dependency installation. The focused checks for blockwise PMI and both diffusion wrappers are in `tests/test_quasid_v5.py`; run them in the target environment before launching experiments.
