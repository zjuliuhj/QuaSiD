"""Regressions for validation conditions and forecast completion detection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from engine.solver import Trainer
from models.quasid.quasid_cont import QuaSiD_cont
from models.quasid.quasid_disc import QuaSiD_disc

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_cls", [QuaSiD_cont, QuaSiD_disc])
@pytest.mark.parametrize("domain", ["frequency", "embedding"])
def test_validation_preserves_conditions_and_training_still_drops_them(monkeypatch, model_cls, domain):
    model = model_cls(
        seq_length=16, channels=2, emb_size=32, num_layers=1, corr_rank=3,
        max_step=8, sampling_steps=2, condition_domain=domain,
        condition_dim=5 if domain == "frequency" else 6,
        device="cpu", attn_backend="sdpa",
    )
    target = torch.randn(3, 16, 2)
    condition = torch.randn(3, 8, 2) if domain == "frequency" else torch.randn(3, 6)
    seen = []
    dropout_draws = []
    original_rand = torch.rand

    def force_dropout(*args, **kwargs):
        if args == (1,):
            dropout_draws.append(True)
            return torch.zeros(1)
        return original_rand(*args, **kwargs)

    def predict_noise(real, imaginary, *args, **kwargs):
        seen.append(kwargs["condition"])
        return torch.zeros_like(real), torch.zeros_like(imaginary)

    monkeypatch.setattr(torch, "rand", force_dropout)
    monkeypatch.setattr(model, "_predict_noise", predict_noise)
    model.eval()
    with torch.inference_mode():
        assert torch.isfinite(model((target, condition)))
    assert seen[-1] is not None
    assert dropout_draws == []
    model.train()
    assert torch.isfinite(model((target, condition)))
    assert seen[-1] is None
    assert dropout_draws == [True]
    assert model._training_condition(None) is None


@pytest.mark.parametrize(
    "completed_step,overrides,should_reuse",
    [(None, [], False), (2000, [], False), (20000, [], True),
     (20000, ["solver.train_steps", "22000"], False),
     (3000, ["solver.train_steps", "3000"], True),
     ("early", [], True),
     ("early", ["solver.validation.early_stopping.patience", "3"], False),
     ("early", ["solver.validation.early_stopping.enabled", "false"], False),
     ("early", ["solver.validation.interval", "1000"], False),
     ("early", ["solver.train_steps", "22000"], False)],
)
def test_best_checkpoint_requires_completed_training(tmp_path, completed_step, overrides, should_reuse):
    script = tmp_path / "scripts/experiments/train_eval.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/experiments/train_eval.sh", script)
    # The completion check must use the same YAML override implementation as training.
    shutil.copytree(ROOT / "utils", tmp_path / "utils", ignore=shutil.ignore_patterns("__pycache__"))
    config = tmp_path / "config/forecasting/etth1.yaml"
    config.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "config/forecasting/etth1.yaml", config)
    checkpoint_dir = tmp_path / "experiments/forecast/quasid_cont/etth1/existing/ckpt"
    checkpoint_dir.mkdir(parents=True)
    # An early best can exist in both incomplete and complete runs.
    torch.save({"step": 2000, "model": {}}, checkpoint_dir / "ckpt-best.pt")
    if completed_step is not None:
        marker = {"step": completed_step}
        if completed_step == "early":
            marker = dict(step=6000, reason="early_stopping", train_steps=20000, patience=2,
                          validation_interval=2000, validation_max_batches=None)
        (checkpoint_dir / "training-complete.json").write_text(json.dumps(marker))
    capture = tmp_path / "commands.txt"
    python = tmp_path / "capture-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "with open(os.environ['CAPTURE'], 'a') as output:\n"
        "    output.write(' '.join(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    environment = dict(
        os.environ, MODE="forecast", DATASET="etth1", MODEL_NAME="quasid_cont",
        EXP_NAME="existing", RUN_TRAIN="true", RUN_EVAL="false", MILESTONE="best",
        TENSORBOARD="false", PYTHON_BIN=str(python), CAPTURE=str(capture),
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(["bash", str(script), *overrides], cwd=tmp_path,
                            env=environment, text=True, capture_output=True, timeout=30)
    assert not capture.exists(), "Existing checkpoints must never be overwritten by fresh training."
    if should_reuse:
        assert result.returncode == 0, result.stderr
        assert "Reuse trained forecast model" in result.stdout
    else:
        assert result.returncode != 0
        assert "Incomplete forecast" in result.stderr
        assert "Reuse trained forecast model" not in result.stdout


class TinyLoss(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.failure = failure

    def forward(self, data):
        if self.failure == "forward":
            raise RuntimeError("interrupted training")
        if self.failure == "validation" and not self.training:
            raise RuntimeError("interrupted validation")
        return self.weight.square() * data.mean()


@pytest.mark.parametrize("failure", [None, "forward", "validation"])
def test_completion_record_only_survives_successful_training(tmp_path, failure):
    marker = tmp_path / "training-complete.json"
    marker.write_text(json.dumps({"step": 20000}))
    config = {
        "solver": {
            "train_steps": 2, "save_cycle": 1, "grad_accum_every": 1,
            "ema": {"decay": 0.995, "update_interval": 1},
            "scheduler": {"target": "engine.lr_sch.CosineAnnealingLRWithWarmup",
                          "params": {"T_max": 2}},
        },
        "dataloader": {"batch_size": 2},
    }
    args = SimpleNamespace(gpu=None, mode="forecast", ckpt_folder=tmp_path, log_interval=1)
    loader = {"dataloader": [torch.ones(2, 1, 1)]}
    trainer = Trainer(config, args, TinyLoss(failure), loader,
                      val_dataloader=loader if failure == "validation" else None)
    if failure:
        with pytest.raises(RuntimeError, match="interrupted"):
            trainer.train()
        assert not marker.exists()
    else:
        trainer.train()
        assert json.loads(marker.read_text()) == {"step": 2}
