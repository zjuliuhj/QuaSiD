import os
import math
from datetime import datetime

from evaluation.distributed import DistributedForecastCollector
from evaluation.io import append_metrics_csv, update_total_metrics_csv
from utils.metric_conditional import compute_conditional_metrics
from utils.metric_forecasting import compute_probabilistic_forecasting_metrics
from utils.metric_unconditional import compute_unconditional_metrics
from utils.visualization import save_distribution_visualizations, save_forecasting_visualizations


class EvaluationRunner:
    def __init__(self, args, config, trainer, dataloader_info, test_dataloader_info=None, logger=None):
        self.args = args
        self.config = config
        self.trainer = trainer
        self.dataloader_info = dataloader_info
        self.test_dataloader_info = test_dataloader_info
        self.logger = logger

    def log(self, message):
        if self.logger is not None:
            self.logger.log_info(message)

    def metric_base_row(self):
        model_params = self.config["model"]["params"]
        dataset_params = self.config["dataloader"]["dataset"]["params"]
        is_forecast = self.args.mode == "forecast"

        # Forecast-only columns stay empty for generation tasks.
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.args.mode,
            "model_name": self.args.model_name,
            "dataset_name": self.args.dataset_name,
            "experiment": self.args.name,
            "ckpt_experiment": self.args.ckpt_name,
            "milestone": self.args.milestone,
            "seed": self.args.seed,
            "eval_tag": self.args.eval_tag,
            "seq_length": model_params["seq_length"],
            "channels": model_params["channels"],
            "backbone": model_params["backbone"],
            "use_pmi_corr": model_params["use_pmi_corr"],
            "pmi_mode": "blockwise",
            "inst_norm": model_params["inst_norm"],
            "normalization": dataset_params.get("normalization", "none"),
            "freq_keep_ratio": model_params.get("freq_keep_ratio", 1.0),
            "freq_keep_bins": model_params.get("freq_keep_bins"),
            "sampling_steps": model_params["sampling_steps"],
            "history_length": dataset_params["history_length"] if is_forecast else None,
            "pred_length": dataset_params["pred_length"] if is_forecast else None,
            "channel_indep": model_params["channel_indep"] if is_forecast else None,
        }

    def save_metrics(self, metrics):
        if not self.args.is_main_process:
            return None

        row = self.metric_base_row()
        row.update(metrics)
        metrics_path = os.path.join(self.args.experiment_dir, "metrics.csv")
        total_path = os.path.join("experiments", self.args.mode, self.args.model_name, self.args.dataset_name, "total.csv")
        saved_path = append_metrics_csv(metrics_path, row)
        saved_total_path = update_total_metrics_csv(
            total_path,
            row,
            key_columns=("experiment", "milestone", "seed"),
        )
        self._log_tensorboard_metrics(metrics)
        self.log(f"Saved metrics to {saved_path}: {metrics}")
        self.log(f"Updated dataset total metrics at {saved_total_path}")
        return saved_path

    def _log_tensorboard_metrics(self, metrics):
        """Write numeric checkpoint metrics against the checkpoint's train step."""
        if self.logger is None:
            return

        global_step = int(self.trainer.step)
        eval_seed = int(self.args.seed)
        # TensorBoard accepts integer steps only. This companion series uses
        # accumulated training seconds as its x-axis for wall-clock comparisons.
        elapsed_time_step = max(0, int(round(float(self.trainer.elapsed_train_seconds))))
        logged_metric_count = 0
        for name, value in metrics.items():
            # `run_id` is CSV bookkeeping for conditional evaluation rather
            # than a model-quality metric.
            if name == "run_id":
                continue
            try:
                scalar_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar_value):
                self.log(f"Skip non-finite evaluation metric {name}={scalar_value}.")
                continue
            self.logger.add_scalar(
                # Separate repeated evaluation seeds so TensorBoard does not
                # draw several stochastic measurements as one ambiguous curve.
                tag=f"eval/seed_{eval_seed}/{name}",
                scalar_value=scalar_value,
                global_step=global_step,
            )
            self.logger.add_scalar(
                tag=f"eval_elapsed_time_s/seed_{eval_seed}/{name}",
                scalar_value=scalar_value,
                global_step=elapsed_time_step,
            )
            logged_metric_count += 1

        if logged_metric_count:
            self.logger.flush()
            self.log(
                f"Logged {logged_metric_count} evaluation metric(s) to TensorBoard "
                f"at training step {global_step}."
            )

    def evaluate(self):
        raise NotImplementedError


class UnconditionalEvaluator(EvaluationRunner):
    def load_reference(self, dataset):
        return dataset.samples

    def evaluate(self):
        model = self.trainer.ema.ema_model
        eval_sampling_steps = self.config["model"]["params"]["sampling_steps"]
        model.sampling_steps = int(eval_sampling_steps)
        self.log(f"Use evaluation sampling_steps={int(eval_sampling_steps)}")

        dataset = self.dataloader_info["dataset"]
        sample_count = len(dataset)
        if self.args.eval_sample_limit is not None:
            sample_count = min(sample_count, int(self.args.eval_sample_limit))
            self.log(f"Use eval_sample_limit={sample_count}")

        samples = self.trainer.sample(
            num=sample_count,
            size_every=self.config["dataloader"]["sample_size"],
        )

        real_data = self.load_reference(dataset)
        if not self.args.skip_visualization:
            paths = save_distribution_visualizations(
                dataset.unnormalize(real_data),
                dataset.unnormalize(samples),
                self.args.eval_dir,
                compare=self.args.visualization_compare,
                seed=self.args.seed,
            )
            self.log(f"Saved distribution visualizations: {paths}")

        if not self.args.skip_metrics:
            metrics = compute_unconditional_metrics(
                dataset.to_metric_space(real_data),
                dataset.to_metric_space(samples),
                max_samples=self.args.metric_samples,
                discriminative_iterations=self.args.discriminative_iterations,
                predictive_iterations=self.args.predictive_iterations,
                seed=self.args.seed,
                device=getattr(self.args, "device", None),
            )
            self.save_metrics(metrics)


class ConditionalEvaluator(EvaluationRunner):
    def evaluate(self):
        test_dataloader = self.test_dataloader_info["dataloader"]
        test_dataset = self.test_dataloader_info["dataset"]
        sampling_steps = self.config["model"]["params"]["sampling_steps"]

        samples, reals = self.trainer.conditional_sample(
            num=len(test_dataset),
            raw_dataloader=test_dataloader,
            sampling_steps=sampling_steps,
        )

        if not self.args.skip_visualization:
            paths = save_distribution_visualizations(
                reals,
                samples,
                self.args.eval_dir,
                compare=self.args.visualization_compare,
                seed=self.args.seed,
            )
            self.log(f"Saved conditional visualizations: {paths}")
        if not self.args.skip_metrics:
            metrics = compute_conditional_metrics(
                test_dataset.to_metric_space(reals),
                test_dataset.to_metric_space(samples),
            )
            metrics["run_id"] = 0
            self.save_metrics(metrics)


class ForecastingEvaluator(EvaluationRunner):
    def evaluate(self):
        test_dataloader = self.test_dataloader_info["dataloader"]
        test_dataset = self.test_dataloader_info["dataset"]
        sampling_steps = self.config["model"]["params"]["sampling_steps"]
        sample_count = len(test_dataset)
        if self.args.eval_sample_limit is not None:
            sample_count = min(sample_count, int(self.args.eval_sample_limit))
            self.log(f"Use forecasting eval_sample_limit={sample_count}")

        target_ensemble, target_reals = self.trainer.conditional_ensemble_sample(
            num=sample_count,
            raw_dataloader=test_dataloader,
            ensemble_size=self.args.num_samples,
            ensemble_chunk_size=self.args.ensemble_chunk_size,
            sampling_steps=sampling_steps,
        )

        pred_length = int(self.config["dataloader"]["dataset"]["params"]["pred_length"])
        reals = target_reals[:, -pred_length:, :]
        sample_ensemble = target_ensemble[:, :, -pred_length:, :]
        contexts = target_reals[:, :-pred_length, :] if target_reals.shape[1] > pred_length else None

        collector = DistributedForecastCollector(self.args)
        reals, sample_ensemble, contexts = collector.collect(reals, sample_ensemble, contexts, sample_count)
        if getattr(self.args, "distributed", False) and not self.args.is_main_process:
            return

        if not self.args.skip_visualization:
            paths = save_forecasting_visualizations(
                reals,
                sample_ensemble,
                self.args.eval_dir,
                contexts=contexts,
                compare=min(4, self.args.visualization_compare),
                seed=self.args.seed,
            )
            self.log(f"Saved forecasting visualizations: {paths}")

        if not self.args.skip_metrics:
            metrics = compute_probabilistic_forecasting_metrics(reals, sample_ensemble)
            self.save_metrics(metrics)


def build_evaluator(args, config, trainer, dataloader_info, test_dataloader_info=None, logger=None):
    runner_cls = {
        "uncond": UnconditionalEvaluator,
        "cond": ConditionalEvaluator,
        "forecast": ForecastingEvaluator,
    }.get(args.mode)
    if runner_cls is None:
        raise ValueError(f"Unrecognized mode: {args.mode}. Please check the input arguments.")
    return runner_cls(args, config, trainer, dataloader_info, test_dataloader_info, logger)
