import pytest
import torch

from models.quasid.backbones import build_backbone
from models.quasid.dit_blocks import SharedQuadraturePMIBlock
from models.quasid.dit_solver_v5 import DiTSolverV5, DiTSolverV5Layer
from models.quasid.quasid_disc import QuaSiD_disc
from models.quasid.quasid_cont import QuaSiD_cont


@pytest.mark.parametrize("gate", range(6))
def test_six_gates_route_shared_quadrature_encodings(gate):
    torch.manual_seed(7)
    block = SharedQuadraturePMIBlock(7, 8, corr_rank=2)
    with torch.no_grad():
        block.time_mlp[-1].bias[gate * 2:(gate + 1) * 2] = 1
    x_r, x_i = torch.randn(2, 7, 3), torch.randn(2, 7, 3)
    actual_r, actual_i = block(x_r, x_i, torch.randn(2, 8))
    if gate in (0, 3):
        encoded = block.b_r(x_r.transpose(1, 2))
    elif gate in (1, 4):
        encoded = block.b_i(x_i.transpose(1, 2))
    else:
        encoded = torch.ones(2, 3, 2)
    decoder = block.a_r if gate < 3 else block.a_i
    expected = decoder(encoded).transpose(1, 2)
    active, inactive = (actual_r, actual_i) if gate < 3 else (actual_i, actual_r)
    torch.testing.assert_close(active, expected)
    assert torch.count_nonzero(inactive) == 0


def test_encoders_are_evaluated_once_and_shared_by_both_outputs():
    block = SharedQuadraturePMIBlock(7, 8, corr_rank=2)
    calls = {"r": 0, "i": 0}
    def count(name):
        def hook(*args):
            calls[name] += 1
        return hook
    handles = [
        block.b_r.register_forward_hook(count("r")),
        block.b_i.register_forward_hook(count("i")),
    ]
    block(torch.randn(2, 7, 3), torch.randn(2, 7, 3), torch.randn(2, 8))
    for handle in handles:
        handle.remove()
    assert calls == {"r": 1, "i": 1}
    assert block.time_mlp[-1].out_features == 12


def test_correction_is_affine_at_fixed_context_and_masks_imaginary_endpoints():
    torch.manual_seed(5)
    block = SharedQuadraturePMIBlock(7, 8, corr_rank=2)
    with torch.no_grad():
        block.time_mlp[-1].bias.normal_()
    a_r, a_i, b_r, b_i = (torch.randn(2, 7, 3) for _ in range(4))
    context = torch.randn(2, 8)
    mask = torch.ones(1, 7, 1)
    mask[:, (0, -1)] = 0
    at_a = block(a_r, a_i, context, mask)
    at_b = block(b_r, b_i, context, mask)
    at_mix = block(0.3 * a_r + 0.7 * b_r, 0.3 * a_i + 0.7 * b_i, context, mask)
    for mixed, first, second in zip(at_mix, at_a, at_b):
        torch.testing.assert_close(mixed, 0.3 * first + 0.7 * second)
    assert torch.count_nonzero(at_a[1][:, (0, -1)]) == 0


@pytest.mark.parametrize("pmi_mode", ["blockwise", "independent"])
def test_backbone_initialization_keeps_all_pmi_blocks_exactly_zero(pmi_mode):
    model = build_backbone(
        "dit_solver_v5", channels=2, emb_size=32, spectrum_bins=7,
        num_layers=2, corr_rank=2, attn_backend="sdpa", pmi_mode=pmi_mode,
    )
    assert isinstance(model, DiTSolverV5)
    blocks = [module for module in model.modules() if isinstance(module, SharedQuadraturePMIBlock)]
    assert len(blocks) == (2 if pmi_mode == "blockwise" else 1)
    for block in blocks:
        corrections = block(torch.randn(2, 7, 2), torch.randn(2, 7, 2), torch.randn(2, 32))
        assert all(torch.count_nonzero(value) == 0 for value in corrections)


def test_v5_pmi_receives_condition_context():
    layer = DiTSolverV5Layer(32, 7, corr_rank=2, attn_backend="sdpa")
    received = []
    handle = layer.pmi.register_forward_pre_hook(lambda module, args: received.append(args[2].detach()))
    c_r, c_i = torch.randn(2, 32), torch.randn(2, 32)
    variance = (torch.ones(1, 7, 1), torch.ones(1, 7, 1))
    layer(torch.randn(2, 7, 32), torch.randn(2, 7, 32), c_r, c_i, torch.ones(2, 1, 1), variance)
    handle.remove()
    torch.testing.assert_close(received[0], (c_r + c_i) * 0.5)


@pytest.mark.parametrize("model_cls", [QuaSiD_cont, QuaSiD_disc])
@pytest.mark.parametrize("pmi_mode", ["blockwise", "independent"])
@pytest.mark.parametrize("case", ["even", "odd", "cropped", "embedding", "forecast", "forecast_ci"])
def test_v5_wrappers_train_and_sample_across_task_contracts(model_cls, case, pmi_mode):
    torch.manual_seed(9)
    conditional = case in {"embedding", "forecast", "forecast_ci"}
    forecast = case.startswith("forecast")
    length = 25 if case == "odd" else 24
    model = model_cls(
        seq_length=length, channels=2, max_step=8, sampling_steps=2,
        backbone="dit_solver_v5", use_pmi_corr=True, corr_rank=3, pmi_mode=pmi_mode,
        emb_size=32, num_layers=2, device="cpu", attn_backend="sdpa",
        freq_keep_ratio=0.5 if case == "cropped" else 1.0,
        condition_domain="frequency" if forecast else "embedding",
        condition_dim=5 if forecast else 6,
        channel_indep=case == "forecast_ci", inst_norm=forecast, cfg_scale=1.5,
    )
    targets = torch.randn(2, length, 2)
    condition = torch.randn(2, 8, 2) if forecast else torch.randn(2, 6)
    loss = model((targets, condition) if conditional else targets)
    assert torch.isfinite(loss)
    loss.backward()
    pmi_gradients = [
        module.time_mlp[-1].weight.grad for module in model.model.modules()
        if isinstance(module, SharedQuadraturePMIBlock)
    ]
    assert pmi_gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in pmi_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in pmi_gradients)
    model.eval()
    if conditional:
        sample = model.generate_conditional(cond=condition, sampling_steps=2)
    else:
        sample = model.generate_unconditional(batch_size=2)
    assert sample.shape == targets.shape
    assert torch.isfinite(sample).all()


@pytest.mark.parametrize("pmi_mode", ["blockwise", "independent"])
def test_v5_without_pmi_uses_independent_branches(pmi_mode):
    model = build_backbone(
        "dit_solver_v5", channels=2, emb_size=32, spectrum_bins=7,
        num_layers=2, use_pmi_corr=False, attn_backend="sdpa", pmi_mode=pmi_mode,
    )
    real, imaginary = torch.randn(2, 7, 2), torch.randn(2, 7, 2)
    first = model(real, imaginary, torch.ones(2))
    second = model(real, imaginary + 1, torch.ones(2))
    assert all(layer.pmi is None for layer in model.layers)
    assert model.pmi is None
    torch.testing.assert_close(first[0], second[0])


def test_independent_pmi_uses_raw_spectrum_and_fuses_after_backbone_heads():
    torch.manual_seed(11)
    params = dict(channels=2, emb_size=32, spectrum_bins=7, num_layers=2,
                  corr_rank=3, attn_backend="sdpa", pmi_mode="independent")
    model = build_backbone("dit_solver_v5", **params)
    base = build_backbone("dit_solver_v5", **params, use_pmi_corr=False)
    base.load_state_dict({key: value for key, value in model.state_dict().items()
                          if not key.startswith("pmi.")})
    assert all(layer.pmi is None for layer in model.layers)
    with torch.no_grad():
        model.pmi.time_mlp[-1].weight.normal_(std=0.1)
        model.pmi.time_mlp[-1].bias.normal_(std=0.1)

    real, imaginary = torch.randn(2, 7, 2), torch.randn(2, 7, 2)
    time = torch.tensor([2.0, 5.0])
    condition = (torch.randn(2, 32), torch.randn(2, 32))
    scale = torch.tensor([0.2, 0.7])[:, None, None]
    var_r, var_i = torch.ones(1, 7, 1), torch.full((1, 7, 1), 0.5)
    var_i[:, (0, -1)] = 0
    calls = []
    handle = model.pmi.register_forward_hook(lambda module, args, output: calls.append((args, output)))
    predicted = model(real, imaginary, time, condition, scale, (var_r, var_i))
    base_predicted = base(real, imaginary, time, condition)
    assert len(calls) == 1
    inputs, correction = calls[0]
    assert inputs[0] is real and inputs[1] is imaginary
    torch.testing.assert_close(inputs[2], model.time_emb(time) + 0.5 * sum(condition))
    assert torch.count_nonzero(correction[1][:, (0, -1)]) == 0
    for actual, marginal, variance, delta in zip(predicted, base_predicted, (var_r, var_i), correction):
        torch.testing.assert_close(actual, marginal - scale * variance * delta)

    # Changing the marginal network must not change the independent correction.
    with torch.no_grad():
        model.conv_r.weight.add_(0.5)
        model.linear_emb_to_patch_i.weight.mul_(2)
    model(real, imaginary, time, condition, scale, (var_r, var_i))
    handle.remove()
    for first, second in zip(correction, calls[1][1]):
        torch.testing.assert_close(first, second)
    sum(value.square().sum() for value in calls[1][1]).backward()
    assert model.pmi.time_mlp[-1].weight.grad is not None
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not name.startswith("pmi."))


@pytest.mark.parametrize("missing", ["noise_scale", "spectral_variance"])
def test_independent_pmi_requires_diffusion_scaling(missing):
    model = DiTSolverV5(channels=2, emb_size=32, spectrum_bins=7, num_layers=1,
                        pmi_mode="independent", attn_backend="sdpa")
    args = dict(noise_scale=torch.ones(2, 1, 1),
                spectral_variance=(torch.ones(1, 7, 1), torch.ones(1, 7, 1)))
    args.pop(missing)
    with pytest.raises(ValueError, match="requires noise_scale and spectral_variance"):
        model(torch.randn(2, 7, 2), torch.randn(2, 7, 2), torch.ones(2), **args)


def test_invalid_pmi_mode_is_rejected():
    with pytest.raises(ValueError, match="pmi_mode"):
        DiTSolverV5(spectrum_bins=7, pmi_mode="invalid")
