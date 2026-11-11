import argparse
import sys


CONFIG_FOLDERS = {
    "uncond": "unconditional",
    "cond": "conditional",
    "forecast": "forecasting",
}


def build_task_parser(mode, description, default_num_samples=5):
    """Build a QuaSiD checkpoint-evaluation CLI for one task mode.

    The parser owns task-local arguments and forwards them to `main.py` without
    reading saved `.npy` sample artifacts. Shapes and metrics are still owned by
    `evaluation.runners`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-name", required=True, help="Dataset config stem, e.g. stocks or etth1.")
    parser.add_argument("--experiment", required=True, help="Experiment name used for evaluation bookkeeping.")
    parser.add_argument("--ckpt-name", default=None, help="Checkpoint experiment name. Defaults to --experiment.")
    parser.add_argument("--config-file", default=None, help="Config path. Defaults to the task config folder.")
    parser.add_argument("--model-name", default="quasid_cont")
    parser.add_argument("--milestone", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=default_num_samples)
    parser.add_argument("--eval-sample-limit", type=int, default=None)
    parser.add_argument("--metric-samples", type=int, default=3000)
    parser.add_argument("--visualization-compare", type=int, default=3000)
    parser.add_argument("--discriminative-iterations", type=int, default=2000)
    parser.add_argument("--predictive-iterations", type=int, default=5000)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    if mode == "forecast":
        parser.add_argument("--ensemble-chunk-size", type=int, default=4)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Optional config overrides passed to main.py.")
    return parser


def default_config_path(mode, dataset_name):
    """Return the canonical config path for a task mode and dataset stem."""
    return f"config/{CONFIG_FOLDERS[mode]}/{dataset_name}.yaml"


def build_main_argv(mode, args):
    """Translate a task evaluation CLI namespace into `main.py` arguments.

    The output always launches checkpoint evaluation, never training. `condition`,
    `unconditional`, and `forecasting` entrypoints differ only in the forwarded
    mode and forecast ensemble batching knobs.
    """
    config_file = args.config_file or default_config_path(mode, args.dataset_name)
    ckpt_name = args.ckpt_name or args.experiment
    argv = [
        "--mode", mode,
        "--name", args.experiment,
        "--ckpt_name", ckpt_name,
        "--config_file", config_file,
        "--model_name", args.model_name,
        "--dataset_name", args.dataset_name,
        "--milestone", str(args.milestone),
        "--num_samples", str(args.num_samples),
        "--metric_samples", str(args.metric_samples),
        "--visualization_compare", str(args.visualization_compare),
        "--discriminative_iterations", str(args.discriminative_iterations),
        "--predictive_iterations", str(args.predictive_iterations),
    ]
    if args.gpu is not None:
        argv.extend(["--gpu", str(args.gpu)])
    if args.eval_sample_limit is not None:
        argv.extend(["--eval_sample_limit", str(args.eval_sample_limit)])
    if mode == "forecast":
        argv.extend(["--ensemble_chunk_size", str(args.ensemble_chunk_size)])
    if args.skip_metrics:
        argv.append("--skip_metrics")
    if args.skip_visualization:
        argv.append("--skip_visualization")
    argv.extend(args.opts)
    return argv


def run_checkpoint_evaluation(mode, args):
    """Run the current QuaSiD checkpoint evaluator through `main.py`.

    This keeps task evaluation as a thin wrapper around `main.py`
    while preserving the current no-saved-samples evaluation contract.
    """
    from main import main as run_quasid

    original_argv = sys.argv
    sys.argv = ["main.py", *build_main_argv(mode, args)]
    try:
        run_quasid()
    finally:
        sys.argv = original_argv
