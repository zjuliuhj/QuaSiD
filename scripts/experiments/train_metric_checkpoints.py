"""Train missing shared evaluators using an existing unconditional model checkpoint."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.ts2vec.ts2vec import TS2Vec
from utils.metric_common import metric_checkpoint_path, save_metric_checkpoint
from utils.metric_discriminative import discriminative_score_metrics
from utils.metric_predictive import predictive_score_metrics
from utils.runtime import instantiate_from_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=os.environ.get("DATASET"))
    parser.add_argument("--experiment-dir", type=Path, default=os.environ.get("EXPERIMENT_DIR"))
    parser.add_argument("--milestone", type=int, default=10)
    parser.add_argument("--samples", type=int, default=3000)
    args = parser.parse_args()
    if not args.dataset or args.experiment_dir is None or args.samples <= 0:
        parser.error("Provide --dataset, --experiment-dir, and a positive --samples value.")
    if not torch.cuda.is_available():
        raise RuntimeError("Metric checkpoint training requires an available CUDA/PPU device.")

    device = torch.device("cuda:0")
    seed_everything(123)
    config_path = args.experiment_dir / "log/configs/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    dataset_config = config["dataloader"]["dataset"]
    dataset_config["params"]["period"] = "train"
    dataset = instantiate_from_config(dataset_config)
    count = min(args.samples, len(dataset))
    rng = np.random.default_rng(123)
    indices = rng.permutation(len(dataset))[:count]
    real = dataset.to_metric_space(dataset.samples[indices])
    cache = ROOT / "experiments/metric_checkpoints"
    paths = {
        metric: metric_checkpoint_path(cache, metric, real, args.dataset)
        for metric in ("discriminator", "predictor", "ts2vec")
    }
    missing = [metric for metric, path in paths.items() if not path.is_file()]
    print(f"{args.dataset} horizon={real.shape[1]} samples={count} missing={missing}", flush=True)
    if not missing:
        return

    if "ts2vec" in missing:
        iterations = 200 if real.size <= 100000 else 600
        print(f"Train TS2Vec: {iterations} steps", flush=True)
        encoder = TS2Vec(
            input_dims=real.shape[-1], device=device, batch_size=8,
            lr=0.001, output_dims=320, max_train_length=3000,
        )
        encoder.fit(real, n_iters=iterations, verbose=True)
        assert encoder.n_iters == iterations
        save_metric_checkpoint(encoder.net, paths["ts2vec"])
        del encoder

    if "discriminator" in missing or "predictor" in missing:
        model_config = config["model"]
        model_config["params"]["device"] = str(device)
        model = instantiate_from_config(model_config).eval()
        checkpoint_path = args.experiment_dir / "ckpt" / f"ckpt-{args.milestone}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        print(
            f"Generate {count} samples from {checkpoint_path}; "
            f"sampler={model_config['params']['sampler']} sampling_steps={model.sampling_steps}",
            flush=True,
        )
        generated = []
        batch_size = config["dataloader"]["sample_size"]
        precision = config["solver"].get("mixed_precision", "no")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=precision in {"fp16", "bf16"},
        ):
            for start in range(0, count, batch_size):
                samples = model.generate_unconditional(batch_size=min(batch_size, count - start))
                generated.append(samples.float().cpu().numpy())
                print(f"Generated {min(start + batch_size, count)}/{count}", flush=True)
        fake = dataset.to_metric_space(np.concatenate(generated, axis=0))
        fake = fake[rng.permutation(count)]
        del model, generated

        if "discriminator" in missing:
            print("Train DS: 2000 steps", flush=True)
            discriminative_score_metrics(
                real, fake, device=device, checkpoint_dir=cache, dataset_name=args.dataset,
            )
        if "predictor" in missing:
            print("Train PS: 5000 steps", flush=True)
            predictive_score_metrics(
                real, fake, device=device, checkpoint_dir=cache, dataset_name=args.dataset,
            )

    for path in paths.values():
        weights = torch.load(path, map_location="cpu", weights_only=True)
        assert weights, path
    print(f"DONE: {args.dataset} horizon={real.shape[1]} all three checkpoints saved", flush=True)


if __name__ == "__main__":
    main()
