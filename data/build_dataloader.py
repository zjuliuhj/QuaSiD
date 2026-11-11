import torch
import copy
from torch.utils.data.distributed import DistributedSampler

from utils.runtime import instantiate_from_config


def _build_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    distributed=False,
    rank=0,
    world_size=1,
):
    """Create a DataLoader and optional DistributedSampler from one dataset object."""
    sampler = None
    global_batch_size = int(batch_size)
    local_batch_size = global_batch_size
    if distributed:
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"DDP expects dataloader.batch_size to be the global batch size and divisible by "
                f"world_size. Got batch_size={global_batch_size}, world_size={world_size}."
            )
        local_batch_size = global_batch_size // world_size
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=shuffle,
        )

    loader_kwargs = {
        "batch_size": local_batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "sampler": sampler,
        "drop_last": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    batch_info = {
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "world_size": world_size if distributed else 1,
    }
    return torch.utils.data.DataLoader(dataset, **loader_kwargs), sampler, batch_info


def _dataset_config_for_period(dataloader_config, period):
    """Runtime split selector for the shared YAML dataset block."""
    dataset_config = copy.deepcopy(dataloader_config["dataset"])
    dataset_config["params"]["period"] = period
    return dataset_config


def build_dataloader(config, args=None):
    """Build the unconditional training loader from the shared time-series dataset config."""
    dataloader_config = config["dataloader"]
    dataset_config = _dataset_config_for_period(dataloader_config, "train")
    dataset = instantiate_from_config(dataset_config)

    distributed = bool(getattr(args, "distributed", False)) and bool(getattr(args, "train", False))
    dataloader, sampler, batch_info = _build_loader(
        dataset=dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config["pin_memory"],
        persistent_workers=dataloader_config["persistent_workers"],
        prefetch_factor=dataloader_config["prefetch_factor"],
        distributed=distributed,
        rank=int(getattr(args, "rank", 0)),
        world_size=int(getattr(args, "world_size", 1)),
    )
    return {
        "dataloader": dataloader,
        "dataset": dataset,
        "sampler": sampler,
        "batch_info": batch_info,
    }


def build_dataloader_conditional(config, args=None, period="train"):
    """Build conditional train/test loaders while keeping data params in one YAML block."""
    if period not in {"train", "test"}:
        raise ValueError(f"Expected period to be 'train' or 'test', got {period!r}.")

    dataloader_config = config["dataloader"]
    dataset_config = _dataset_config_for_period(dataloader_config, period)

    dataset = instantiate_from_config(dataset_config)

    is_train = period == "train"
    batch_size = dataloader_config["batch_size"] if is_train else dataloader_config["sample_size"]
    distributed = bool(getattr(args, "distributed", False)) and is_train
    dataloader, sampler, batch_info = _build_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config["pin_memory"],
        persistent_workers=dataloader_config["persistent_workers"],
        prefetch_factor=dataloader_config["prefetch_factor"],
        distributed=distributed,
        rank=int(getattr(args, "rank", 0)),
        world_size=int(getattr(args, "world_size", 1)),
    )
    return {
        "dataloader": dataloader,
        "dataset": dataset,
        "sampler": sampler,
        "batch_info": batch_info,
    }


def build_dataloader_forecasting(config, args=None, period="train"):
    """Build forecasting dataloaders from one shared BLC dataset config."""
    if period not in {"train", "val", "test"}:
        raise ValueError(f"Expected period to be 'train', 'val', or 'test', got {period!r}.")

    dataloader_config = config["dataloader"]
    dataset_config = _dataset_config_for_period(dataloader_config, period)

    # Forecast evaluation can cap the number of test windows without changing data params.
    if period == "test":
        sample_limit = getattr(args, "eval_sample_limit", None) if args is not None else None
        sample_limit = dataloader_config["test_sample_limit"] if sample_limit is None else sample_limit
        if sample_limit is not None:
            dataset_config["params"]["sample_limit"] = int(sample_limit)

    dataset = instantiate_from_config(dataset_config)

    is_train = period == "train"
    batch_size = dataloader_config["batch_size"] if is_train else dataloader_config["sample_size"]
    distributed = bool(getattr(args, "distributed", False)) and is_train
    dataloader, sampler, batch_info = _build_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=dataloader_config["num_workers"],
        pin_memory=dataloader_config["pin_memory"],
        persistent_workers=dataloader_config["persistent_workers"],
        prefetch_factor=dataloader_config["prefetch_factor"],
        distributed=distributed,
        rank=int(getattr(args, "rank", 0)),
        world_size=int(getattr(args, "world_size", 1)),
    )
    return {
        "dataloader": dataloader,
        "dataset": dataset,
        "sampler": sampler,
        "batch_info": batch_info,
    }
