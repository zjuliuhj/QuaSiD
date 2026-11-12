import pytest
import torch

from models.quasid.backbones import build_backbone
from models.quasid.dit_blocks import SharedQuadraturePMIBlock
from models.quasid.dit_solver_v5 import DiTSolver, DiTSolverLayer
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


def test_backbone_initialization_keeps_all_pmi_blocks_exactly_zero():
    model = build_backbone(
        "dit_solver_v5", channels=2, emb_size=32, spectrum_bins=7,
        num_layers=2, corr_rank=2, attn_backend="sdpa",
    )
    assert isinstance(model, DiTSolver)
    blocks = [module for module in model.modules() if isinstance(module, SharedQuadraturePMIBlock)]
    assert len(blocks) == 2
    for block in blocks:
        corrections = block(torch.randn(2, 7, 2), torch.randn(2, 7, 2), torch.randn(2, 32))
        assert all(torch.count_nonzero(value) == 0 for value in corrections)


def test_v5_pmi_receives_condition_context():
    layer = DiTSolverLayer(32, 7, corr_rank=2, attn_backend="sdpa")
    received = []
    handle = layer.pmi.register_forward_pre_hook(lambda module, args: received.append(args[2].detach()))
    c_r, c_i = torch.randn(2, 32), torch.randn(2, 32)
    variance = (torch.ones(1, 7, 1), torch.ones(1, 7, 1))
    layer(torch.randn(2, 7, 32), torch.randn(2, 7, 32), c_r, c_i, torch.ones(2, 1, 1), variance)
    handle.remove()
    torch.testing.assert_close(received[0], (c_r + c_i) * 0.5)


@pytest.mark.parametrize("model_cls", [QuaSiD_cont, QuaSiD_disc])
@pytest.mark.parametrize("case", ["even", "odd", "cropped", "embedding", "forecast", "forecast_ci"])
def test_v5_wrappers_train_and_sample_across_task_contracts(model_cls, case):
    torch.manual_seed(9)
    conditional = case in {"embedding", "forecast", "forecast_ci"}
    forecast = case.startswith("forecast")
    length = 25 if case == "odd" else 24
    model = model_cls(
        seq_length=length, channels=2, max_step=8, sampling_steps=2,
        sampler="ode",
        backbone="dit_solver_v5", use_pmi_corr=True, corr_rank=3,
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


def test_v5_without_pmi_uses_independent_branches():
    model = build_backbone(
        "dit_solver_v5", channels=2, emb_size=32, spectrum_bins=7,
        num_layers=2, use_pmi_corr=False, attn_backend="sdpa",
    )
    real, imaginary = torch.randn(2, 7, 2), torch.randn(2, 7, 2)
    first = model(real, imaginary, torch.ones(2))
    second = model(real, imaginary + 1, torch.ones(2))
    assert all(layer.pmi is None for layer in model.layers)
    torch.testing.assert_close(first[0], second[0])


@pytest.mark.parametrize("missing", ["noise_scale", "spectral_variance"])
def test_v5_pmi_requires_diffusion_scaling(missing):
    model = DiTSolver(channels=2, emb_size=32, spectrum_bins=7, num_layers=1,
                      attn_backend="sdpa")
    args = dict(noise_scale=torch.ones(2, 1, 1),
                spectral_variance=(torch.ones(1, 7, 1), torch.ones(1, 7, 1)))
    args.pop(missing)
    with pytest.raises(ValueError, match="requires noise_scale and spectral_variance"):
        model(torch.randn(2, 7, 2), torch.randn(2, 7, 2), torch.ones(2), **args)


@pytest.mark.parametrize("num_steps", [1, 3])
def test_cont_sde_omits_noise_only_on_last_step(monkeypatch, num_steps):
    model = QuaSiD_cont(
        seq_length=24, channels=2, emb_size=32, num_layers=1,
        device="cpu", attn_backend="sdpa",
    )
    monkeypatch.setattr(
        model, "_predict_noise",
        lambda x_r, x_i, *args, **kwargs: (torch.zeros_like(x_r), torch.zeros_like(x_i)),
    )
    states = []
    original_step = model._euler_maruyama_step

    def record_step(x_r, x_i, t_value, dt, stats, **kwargs):
        states.append((x_r.clone(), x_i.clone(), t_value, dt, kwargs["add_noise"]))
        return original_step(x_r, x_i, t_value, dt, stats, **kwargs)

    monkeypatch.setattr(model, "_euler_maruyama_step", record_step)
    torch.manual_seed(41)
    actual_r, actual_i, stats = model._sample_spectrum(2, num_steps)
    assert [state[4] for state in states] == [True] * (num_steps - 1) + [False]

    last_r, last_i, t_value, dt, _ = states[-1]
    beta = model.get_beta(torch.tensor(t_value))
    torch.testing.assert_close(actual_r, last_r - 0.5 * beta * last_r * dt)
    torch.testing.assert_close(actual_i, (last_i - 0.5 * beta * last_i * dt) * (stats["var_i"] > 0))


@pytest.mark.parametrize("length,num_steps", [(24, 1), (25, 200)])
def test_cont_ode_recovers_clean_spectrum_with_exact_noise(monkeypatch, length, num_steps):
    model = QuaSiD_cont(
        seq_length=length, channels=2, emb_size=32, num_layers=1,
        sampler="ode", device="cpu", attn_backend="sdpa",
    )
    clean_r, clean_i, stats = model._initial_spectrum_state(2)
    calls = []

    def exact_noise(x_r, x_i, t, stats, **kwargs):
        calls.append(t.clone())
        mean, noise_scale = model.get_marginal_prob(t)
        return (x_r - mean * clean_r) / noise_scale, (x_i - mean * clean_i) / noise_scale

    monkeypatch.setattr(model, "_predict_noise", exact_noise)
    actual_r, actual_i, _ = model._sample_spectrum(2, num_steps)
    assert len(calls) == num_steps
    assert all(torch.all(t > 0) for t in calls)
    torch.testing.assert_close(actual_r, clean_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual_i, clean_i, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(actual_i[:, 0]) == 0


@pytest.mark.parametrize("sampler", ["ode", "sde"])
def test_disc_sampler_controls_step_noise(monkeypatch, sampler):
    model = QuaSiD_disc(
        seq_length=24, channels=2, max_step=8, emb_size=32, num_layers=1,
        sampler=sampler, device="cpu", attn_backend="sdpa",
    )
    x_r, x_i, stats = model._initial_spectrum_state(2)
    monkeypatch.setattr(
        model, "_predict_noise",
        lambda x_r, x_i, *args, **kwargs: (torch.zeros_like(x_r), torch.zeros_like(x_i)),
    )
    torch.manual_seed(23)
    before = torch.get_rng_state()
    first = model._ddpm_step(x_r, x_i, 7, 3, stats)
    after = torch.get_rng_state()
    second = model._ddpm_step(x_r, x_i, 7, 3, stats)
    assert torch.equal(before, after) == (sampler == "ode")
    assert torch.equal(first[0], second[0]) == (sampler == "ode")
    if sampler == "ode":
        alpha = torch.sqrt(model.ddpm.alpha_bar[3] / model.ddpm.alpha_bar[7])
        torch.testing.assert_close(first[0], alpha * x_r)
    assert torch.count_nonzero(first[1][:, (0, -1)]) == 0

    before_terminal = torch.get_rng_state()
    terminal_r, terminal_i = model._ddpm_step(x_r, x_i, 0, -1, stats)
    assert torch.equal(before_terminal, torch.get_rng_state())
    torch.testing.assert_close(terminal_r, x_r / torch.sqrt(model.ddpm.alpha_bar[0]))
    assert torch.count_nonzero(terminal_i[:, (0, -1)]) == 0


@pytest.mark.parametrize("model_cls", [QuaSiD_cont, QuaSiD_disc])
def test_wrappers_reject_unknown_sampler(model_cls):
    with pytest.raises(ValueError, match="sampler must be 'sde' or 'ode'"):
        model_cls(seq_length=24, channels=2, sampler="ddim", device="cpu")
