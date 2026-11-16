import copy
import importlib
import random
import sys
import warnings

import numpy as np
import torch
import yaml


def load_yaml_config(path):
    with open(path, encoding="utf-8-sig") as f:
        return yaml.full_load(f)


def save_config_to_yaml(config, path):
    assert path.endswith(".yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def merge_opts_to_config(config, opts):
    """Merge trailing CLI overrides into a YAML config with YAML scalar typing."""
    def parse_value(value):
        if not isinstance(value, str):
            return value
        return yaml.safe_load(value)

    def cast_value(current_value, value):
        value = parse_value(value)
        if current_value is None:
            return value
        if isinstance(current_value, bool):
            if isinstance(value, bool):
                return value
            value_lower = str(value).lower()
            if value_lower in {"true", "1", "yes", "y", "on"}:
                return True
            if value_lower in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError(f"Cannot parse boolean override value: {value}")
        if isinstance(value, type(current_value)):
            return value
        return type(current_value)(value)

    def modify_dict(current, path, value):
        if len(path) == 1:
            current[path[0]] = cast_value(current[path[0]], value) if path[0] in current else parse_value(value)
        else:
            if path[0] not in current:
                current[path[0]] = {}
            if not isinstance(current[path[0]], dict):
                raise KeyError(f"Cannot set nested override under non-dict key: {path[0]}")
            current[path[0]] = modify_dict(current[path[0]], path[1:], value)
        return current

    if opts is not None and len(opts) > 0:
        opts = list(opts)
        if opts[0] == "--":
            opts = opts[1:]
        if len(opts) % 2 != 0:
            raise ValueError("Each override should be a name/value pair.")
        for i in range(len(opts) // 2):
            name = opts[2 * i]
            value = opts[2 * i + 1]
            config = modify_dict(config, name.split("."), value)
    return config



def forecast_condition_dim(dataset_params, condition_domain):
    """Derive per-channel forecasting condition width from dataset contract.

    Frequency conditions use rFFT bins `history_length // 2 + 1`; time-domain
    conditions use the raw `history_length`. Channels are handled separately by
    the model-level channel injection path, so this width never includes `C`.
    """
    history_length = int(dataset_params["history_length"])
    if condition_domain == "frequency":
        return history_length // 2 + 1
    if condition_domain == "time":
        return history_length
    raise ValueError(f"Unknown condition_domain={condition_domain!r}; use 'time' or 'frequency'.")


def materialize_runtime_config(config, mode):
    """Return a config copy with derived runtime-only model parameters.

    User YAML stays free of shape-derived fields. Forecasting injects
    `condition_dim` only immediately before model construction because the
    current QuaSiD condition encoder needs a fixed per-channel input width.
    """
    runtime_config = copy.deepcopy(config)
    if mode == "forecast":
        model_params = runtime_config["model"]["params"]
        dataset_params = runtime_config["dataloader"]["dataset"]["params"]
        model_params["condition_dim"] = forecast_condition_dim(dataset_params, model_params["condition_domain"])
    return runtime_config

def checkpoint_milestone(value):
    """Accept existing positive numeric checkpoint IDs and the validation-best ID."""
    value = str(value)
    if value == "best":
        return value
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("Checkpoint must be a positive integer or 'best'.")
    return int(value)


def validate_config(config, mode):
    """Validate shared YAML invariants, then apply target-specific model checks."""
    top_level = set(config)
    required = {"model", "solver", "dataloader"}
    missing = required - top_level
    if missing:
        raise ValueError(f"Missing top-level config sections: {sorted(missing)}")
    dataloader = config["dataloader"]
    if "train_dataset" in dataloader or "test_dataset" in dataloader:
        raise ValueError("Use dataloader.dataset only; train_dataset/test_dataset were removed.")
    if "dataset" not in dataloader or "params" not in dataloader["dataset"]:
        raise ValueError("dataloader.dataset.params is required by the QuaSiD schema.")
    dataset_params = dataloader["dataset"]["params"]
    if "test_proportion" in dataloader:
        raise ValueError("Use dataloader.dataset.params.train_test_prop for conditional train/test split.")
    if "results_folder" in config["solver"]:
        raise ValueError("solver.results_folder was removed; checkpoints are saved under args.ckpt_folder.")
    if "save2npy" in dataset_params:
        raise ValueError("dataloader.dataset.params.save2npy was removed; evaluation does not dump truth arrays.")

    model_params = config["model"]["params"]
    if mode == "uncond" and bool(model_params.get("inst_norm", False)):
        raise ValueError("unconditional mode requires model.params.inst_norm=false.")

    if "condition_size" in model_params:
        raise ValueError("model.params.condition_size was renamed to condition_dim.")
    if mode == "cond":
        if "proportion" in dataset_params:
            raise ValueError("conditional dataloader.dataset.params.proportion was renamed to train_test_prop.")
        if "train_test_prop" not in dataset_params:
            raise ValueError("conditional dataloader.dataset.params.train_test_prop is required.")
        train_test_prop = float(dataset_params["train_test_prop"])
        if not 0.0 < train_test_prop < 1.0:
            raise ValueError(f"conditional train_test_prop must be in (0, 1), got {train_test_prop}.")
    if mode == "cond" and model_params.get("condition_domain") == "embedding" and "condition_dim" not in model_params:
        raise ValueError("conditional embedding configs must set model.params.condition_dim to the external embedding width.")

    if mode != "forecast":
        return

    if "split_ratio" in dataset_params:
        raise ValueError("forecast dataloader.dataset.params uses train_ratio and val_ratio; split_ratio was removed.")
    validation = config["solver"].get("validation", {})
    if int(validation.get("interval", 1)) <= 0:
        raise ValueError("solver.validation.interval must be positive.")
    if validation.get("max_batches") is not None and int(validation["max_batches"]) <= 0:
        raise ValueError("solver.validation.max_batches must be positive or null.")
    early_stopping = validation.get("early_stopping", {})
    if int(early_stopping.get("patience", 2)) <= 0:
        raise ValueError("solver.validation.early_stopping.patience must be positive.")
    split_lengths = dataset_params.get("split_lengths")
    if split_lengths is not None:
        if len(split_lengths) != 3 or any(type(value) is not int or value <= 0 for value in split_lengths):
            raise ValueError("split_lengths must contain three positive integer train/val/test lengths.")
    else:
        for ratio_name in ("train_ratio", "val_ratio"):
            if ratio_name not in dataset_params:
                raise ValueError(f"forecast dataloader.dataset.params.{ratio_name} is required.")
        train_ratio = float(dataset_params["train_ratio"])
        val_ratio = float(dataset_params["val_ratio"])
        if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
            raise ValueError(
                "forecast train_ratio and val_ratio must be positive and leave a non-empty test split; "
                f"got train_ratio={train_ratio}, val_ratio={val_ratio}."
            )

    history_length = int(dataset_params["history_length"])
    pred_length = int(dataset_params["pred_length"])
    expected_seq_length = history_length + pred_length
    if int(model_params["seq_length"]) != expected_seq_length:
        raise ValueError(
            f"model.params.seq_length must equal history_length + pred_length; "
            f"got {model_params['seq_length']} and {expected_seq_length}."
        )

    if int(model_params["channels"]) != int(dataset_params["channels"]):
        raise ValueError(
            f"model.params.channels must equal dataloader.dataset.params.channels; "
            f"got {model_params['channels']} and {dataset_params['channels']}."
        )

    if "channel_indep" not in model_params:
        raise ValueError("forecast model.params.channel_indep is required.")

    condition_domain = model_params["condition_domain"]
    forecast_condition_dim(dataset_params, condition_domain)
    if "condition_dim" in model_params:
        raise ValueError(
            "forecast model.params.condition_dim is derived from history_length and condition_domain; "
            "remove it from YAML and CLI overrides."
        )

def instantiate_from_config(config):
    if config is None:
        return None
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", {}))


def seed_everything(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn(
            "You have chosen deterministic CUDNN. This can slow down training and "
            "may change restart behavior.",
            stacklevel=2,
        )


def write_args(args, path):
    args_dict = {name: getattr(args, name) for name in dir(args) if not name.startswith("_")}
    with open(path, "a", encoding="utf-8") as args_file:
        args_file.write(f"==> torch version: {torch.__version__}\n")
        args_file.write(f"==> cudnn version: {torch.backends.cudnn.version()}\n")
        args_file.write("==> Cmd:\n")
        args_file.write(str(sys.argv))
        args_file.write("\n==> args:\n")
        for key, value in sorted(args_dict.items()):
            args_file.write(f"  {key}: {value}\n")


def get_model_parameters_info(model):
    parameters = {"overall": {"trainable": 0, "non_trainable": 0, "total": 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {"trainable": 0, "non_trainable": 0}
        for _, param in child_module.named_parameters():
            if param.requires_grad:
                parameters[child_name]["trainable"] += param.numel()
            else:
                parameters[child_name]["non_trainable"] += param.numel()
        parameters[child_name]["total"] = (
            parameters[child_name]["trainable"] + parameters[child_name]["non_trainable"]
        )

        parameters["overall"]["trainable"] += parameters[child_name]["trainable"]
        parameters["overall"]["non_trainable"] += parameters[child_name]["non_trainable"]
        parameters["overall"]["total"] += parameters[child_name]["total"]

    def format_number(num):
        k = 2**10
        m = 2**20
        g = 2**30
        if num > g:
            unit = "G"
            num = round(float(num) / g, 2)
        elif num > m:
            unit = "M"
            num = round(float(num) / m, 2)
        elif num > k:
            unit = "K"
            num = round(float(num) / k, 2)
        else:
            unit = ""
        return f"{num}{unit}"

    def format_dict(values):
        for key, value in values.items():
            if isinstance(value, dict):
                format_dict(value)
            else:
                values[key] = format_number(value)

    format_dict(parameters)
    return parameters
