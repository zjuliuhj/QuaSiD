#!/usr/bin/env python3
"""Summarize repeated checkpoint-evaluation metrics by random seed.

The script keeps the latest row for each requested seed.  This makes an
interrupted evaluation safe to resume: rerunning a seed replaces its value in
the summary rather than double-counting it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = {
    "timestamp",
    "mode",
    "model_name",
    "dataset_name",
    "experiment",
    "ckpt_experiment",
    "milestone",
    "seed",
    "eval_tag",
    "seq_length",
    "channels",
    "backbone",
    "use_pmi_corr",
    "pmi_mode",
    "inst_norm",
    "normalization",
    "freq_keep_ratio",
    "freq_keep_bins",
    "sampling_steps",
    "history_length",
    "pred_length",
    "channel_indep",
    "metric_samples",
    "run_id",
}


def summarize_metrics(metrics_path: Path, experiment: str, milestone: int | str, seeds: list[int]) -> pd.DataFrame:
    """Return mean, sample variance, and sample standard deviation per metric."""
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required to compute sample variance.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    rows = pd.read_csv(metrics_path)
    required_columns = {"experiment", "milestone", "seed"}
    missing_columns = required_columns.difference(rows.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{metrics_path} has no {missing} column. Re-run evaluation with seed recording enabled.")

    rows = rows.copy()
    rows["milestone"] = rows["milestone"].astype(str)
    rows["seed"] = pd.to_numeric(rows["seed"], errors="coerce")
    selected = rows.loc[
        (rows["experiment"] == experiment)
        & (rows["milestone"] == str(milestone))
        & (rows["seed"].isin(seeds))
    ].copy()
    selected = selected.drop_duplicates(subset="seed", keep="last")

    present_seeds = set(selected["seed"].dropna().astype(int))
    missing_seeds = sorted(set(seeds).difference(present_seeds))
    if missing_seeds:
        missing = ", ".join(str(seed) for seed in missing_seeds)
        raise ValueError(
            f"Missing evaluation rows for experiment={experiment!r}, milestone={milestone}, seeds={missing}."
        )

    metric_columns = [
        column
        for column in selected.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(selected[column])
    ]
    if not metric_columns:
        raise ValueError("No numeric metric columns were found.")

    values = selected[metric_columns]
    summary = pd.DataFrame(
        {
            "metric": metric_columns,
            "n": values.count().to_numpy(),
            "mean": values.mean().to_numpy(),
            "variance": values.var(ddof=1).to_numpy(),
            "std": values.std(ddof=1).to_numpy(),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated evaluation metrics across seeds.")
    parser.add_argument("--metrics-path", type=Path, required=True, help="Path to the experiment metrics.csv.")
    parser.add_argument("--output", type=Path, required=True, help="CSV path for the metric summary.")
    parser.add_argument("--experiment", required=True, help="Experiment name to select.")
    parser.add_argument("--milestone", required=True, help="Checkpoint milestone to select.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Exactly the evaluation seeds to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_metrics(args.metrics_path, args.experiment, args.milestone, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.4f")
    print(f"Saved summary for {len(args.seeds)} seeds to {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
