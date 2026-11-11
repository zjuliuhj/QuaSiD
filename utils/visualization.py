from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PNG_DPI = 600
SERIF_STACK = [
    "Times New Roman",
    "STIXGeneral",
    "Nimbus Roman",
    "Liberation Serif",
    "DejaVu Serif",
]
SINGLE_FIGSIZE = (5.4, 3.75)
SQUARE_FIGSIZE = (5.0, 4.75)
PCA_FIGSIZE = (5.35, 5.2)
PANEL_CELL_SIZE = (4.55, 2.95)
OVERVIEW_FIGSIZE = (9.2, 6.4)

# TPAMI QuaSiD Nordic Dusk palette (colorspace-nordic-dusk.json, anchored at #355C7D).
REAL_COLOR = "#235284"          # ocean-navy: primal ground truth / target distribution
REAL_FILL = "#DAE9F8"           # ocean-navy pastel shade
REAL_EDGE = "#6C93BC"           # ocean-navy mid-tone contour edge
GENERATED_COLOR = "#802F7E"     # mulberry-plum: QuaSiD model / generated distribution
GENERATED_FILL = "#F3E3F3"      # mulberry-plum pastel shade
GENERATED_EDGE = "#BA89B9"      # mulberry-plum mid-tone contour edge
CI50_FILL = "#DFC5DF"           # mulberry-plum inner 50% interval shade
HISTORY_COLOR = "#7D6340"       # warm-bronze: conditioning history context
INK_COLOR = "#1E293B"           # primary label/legend ink
MUTED_INK = "#3B4856"           # cool-slate tick text
SPINE_COLOR = "#4E5B68"         # TPAMI QuaSiD coordinate axis slate
BASELINE_COLOR = "#CBD5E1"      # legend frame & subtle guide slate
GRID_COLOR = "#DCE3EC"          # Nordic Dusk subtle grid slate
HORIZON_BG = "#F6F8FB"          # cool-slate pale forecast horizon backdrop
GENERATED_LINESTYLE = (0, (4.8, 2.4))

SPINE_WIDTH = 1.45
TICK_WIDTH = 1.2
CURVE_WIDTH = 2.25
ENVELOPE_EDGE_WIDTH = 1.05

RC_STYLE = {
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": SPINE_WIDTH,
    "axes.labelcolor": INK_COLOR,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "text.color": INK_COLOR,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _as_time_series(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"Expected time-series data with shape (N, L, C), got {arr.shape}")
    return arr


def _paired_subset(real_data, fake_data, compare: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    real_arr = _as_time_series(real_data)
    fake_arr = _as_time_series(fake_data)
    sample_count = min(compare, real_arr.shape[0], fake_arr.shape[0])
    if sample_count <= 1:
        raise ValueError("Need at least two samples for distribution visualization.")

    rng = np.random.default_rng(seed)
    if real_arr.shape[0] == fake_arr.shape[0]:
        # Preserve index alignment when evaluating paired conditional samples.
        idx = rng.permutation(real_arr.shape[0])[:sample_count]
        return real_arr[idx], fake_arr[idx]

    real_idx = rng.permutation(real_arr.shape[0])[:sample_count]
    fake_idx = rng.permutation(fake_arr.shape[0])[:sample_count]
    return real_arr[real_idx], fake_arr[fake_idx]


def _sequence_features(
    real_data, fake_data, compare: int = 3000, seed: int = 123
) -> tuple[np.ndarray, np.ndarray]:
    real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
    return real_sub.mean(axis=2), fake_sub.mean(axis=2)


def _style_axis(ax: plt.Axes, *, grid_axes: str = "both") -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(SPINE_WIDTH)

    if grid_axes in {"both", "x", "y"}:
        ax.grid(axis=grid_axes, color=GRID_COLOR, linewidth=0.8, linestyle="-", alpha=0.85)
    ax.tick_params(
        axis="both",
        which="major",
        colors=SPINE_COLOR,
        labelcolor=MUTED_INK,
        labelsize=10.8,
        length=4.5,
        width=TICK_WIDTH,
        pad=3.5,
    )
    ax.locator_params(axis="both", nbins=5)
    try:
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
    except AttributeError:
        pass

    ax.xaxis.label.set_fontsize(11.8)
    ax.yaxis.label.set_fontsize(11.8)
    ax.xaxis.labelpad = 5
    ax.yaxis.labelpad = 5


def _collect_legend_entries(fig: plt.Figure) -> dict[str, object]:
    entries: dict[str, object] = {}
    for ax in fig.axes:
        if not ax.axison:
            continue
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                entries.setdefault(label, handle)
    return entries


def _attach_in_axis_legend(
    ax: plt.Axes,
    entries: dict[str, object],
    *,
    loc: str = "best",
    ncol: int = 1,
) -> None:
    leg = ax.legend(
        entries.values(),
        entries.keys(),
        loc=loc,
        ncol=min(ncol, len(entries)),
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor=BASELINE_COLOR,
        framealpha=0.90,
        fontsize=9.8,
        labelcolor=INK_COLOR,
        handlelength=2.0,
        handletextpad=0.48,
        columnspacing=0.95,
        borderpad=0.38,
        labelspacing=0.28,
    )
    leg.get_frame().set_linewidth(0.85)
    leg.set_zorder(10)


def _save_figure(
    fig: plt.Figure,
    output_path: str | Path,
    *,
    extra_handles: dict[str, object] | None = None,
    legend_ax: plt.Axes | None = None,
    legend_loc: str = "best",
    legend_ncol: int = 1,
    grid_axes: str | dict[plt.Axes, str] = "both",
    skip_style_axes: set[plt.Axes] | None = None,
) -> str:
    """Export matching PNG and vector PDF files with TPAMI QuaSiD typography."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    skipped = skip_style_axes or set()

    with matplotlib.rc_context(RC_STYLE):
        active_axes: list[plt.Axes] = []
        for ax in fig.axes:
            if not ax.axison or ax in skipped:
                continue
            mode = grid_axes.get(ax, "both") if isinstance(grid_axes, dict) else grid_axes
            _style_axis(ax, grid_axes=mode)
            active_axes.append(ax)

        legend_entries = dict(extra_handles) if extra_handles is not None else _collect_legend_entries(fig)
        target_ax = legend_ax if legend_ax is not None else (active_axes[0] if active_axes else None)
        if legend_entries and target_ax is not None:
            _attach_in_axis_legend(target_ax, legend_entries, loc=legend_loc, ncol=legend_ncol)

        engine = fig.get_layout_engine()
        if engine is not None:
            engine.set(h_pad=0.06, w_pad=0.06)

        try:
            fig.savefig(out, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.04, facecolor="white")
            fig.savefig(
                out.with_suffix(".pdf"),
                dpi=PNG_DPI,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
            )
        finally:
            plt.close(fig)

    return str(out)


def _kde_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if values.size < 2 or np.std(values) < 1e-8:
        values = values + np.random.default_rng(0).normal(0.0, 1e-5, size=values.shape)
    return gaussian_kde(values)(x_grid)


def _add_2d_contour_ring(
    ax: plt.Axes, points: np.ndarray, stroke_color: str, fill_color: str, linestyle
) -> None:
    if points.shape[0] < 25 or np.any(points.std(axis=0) < 1e-6):
        return
    try:
        kde = gaussian_kde(points.T)
    except np.linalg.LinAlgError:
        return

    pad = 0.14 * (points.max(axis=0) - points.min(axis=0) + 1e-6)
    lo = points.min(axis=0) - pad
    hi = points.max(axis=0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 72), np.linspace(lo[1], hi[1], 72))
    zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

    # Outer stroked boundary coincides exactly with the translucent support shell (zero outer halo).
    sample_z = kde(points.T)
    level = float(np.quantile(sample_z, 0.22))
    if np.isfinite(level) and zz.min() < level < zz.max():
        ax.contourf(gx, gy, zz, levels=[level, float(zz.max())], colors=[fill_color], alpha=0.36, zorder=1)
        ax.contour(
            gx,
            gy,
            zz,
            levels=[level],
            colors=[stroke_color],
            linewidths=1.55,
            linestyles=[linestyle],
            alpha=0.92,
            zorder=4,
        )


def _draw_interleaved_embedding(
    ax: plt.Axes, real_points: np.ndarray, fake_points: np.ndarray, seed: int
) -> dict[str, object]:
    n_real = real_points.shape[0]
    n_fake = fake_points.shape[0]
    total = n_real + n_fake

    marker_size = float(np.clip(10500.0 / max(n_real, 160), 13.0, 28.0))
    alpha = float(np.clip(0.80 - 0.18 * np.log10(max(n_real, 100) / 100.0), 0.50, 0.80))

    _add_2d_contour_ring(ax, real_points, REAL_COLOR, REAL_FILL, "-")
    _add_2d_contour_ring(ax, fake_points, GENERATED_COLOR, GENERATED_FILL, GENERATED_LINESTYLE)

    all_points = np.concatenate([real_points, fake_points], axis=0)
    is_fake = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(n_fake, dtype=bool)])
    perm = np.random.default_rng(seed).permutation(total)
    pts = all_points[perm]
    colors = np.where(is_fake[perm], GENERATED_COLOR, REAL_COLOR)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=colors,
        s=marker_size,
        alpha=alpha,
        edgecolors="white",
        linewidths=0.4,
        rasterized=True,
        zorder=3,
    )
    ax.margins(0.08)

    return {
        "Real": Line2D(
            [0], [0], marker="o", linestyle="-", linewidth=1.6, markersize=6.2,
            markerfacecolor=REAL_COLOR, markeredgecolor="white", markeredgewidth=0.6,
            color=REAL_COLOR,
        ),
        "QuaSiD": Line2D(
            [0], [0], marker="o", linestyle=GENERATED_LINESTYLE, linewidth=1.6, markersize=6.2,
            markerfacecolor=GENERATED_COLOR, markeredgecolor="white", markeredgewidth=0.6,
            color=GENERATED_COLOR,
        ),
    }


def save_pca_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
        pca = PCA(n_components=2)
        pca.fit(real_features)
        real_points = pca.transform(real_features)
        fake_points = pca.transform(fake_features)

        fig = plt.figure(figsize=PCA_FIGSIZE, layout="constrained")
        grid = fig.add_gridspec(
            2, 2, width_ratios=(4.6, 1.05), height_ratios=(1.05, 4.6), hspace=0.04, wspace=0.04
        )
        ax = fig.add_subplot(grid[1, 0])
        top = fig.add_subplot(grid[0, 0], sharex=ax)
        right = fig.add_subplot(grid[1, 1], sharey=ax)

        handles = _draw_interleaved_embedding(ax, real_points, fake_points, seed)
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var.)")

        x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
        for dim, marginal, limits in ((0, top, x_limits), (1, right, y_limits)):
            support = np.linspace(*limits, 256)
            for points, stroke, fill, linestyle in (
                (real_points, REAL_COLOR, REAL_FILL, "-"),
                (fake_points, GENERATED_COLOR, GENERATED_FILL, GENERATED_LINESTYLE),
            ):
                density = _kde_curve(points[:, dim], support)
                if dim == 0:
                    marginal.fill_between(support, density, color=fill, alpha=0.62, linewidth=0)
                    marginal.plot(support, density, color=stroke, linewidth=1.9, linestyle=linestyle)
                else:
                    marginal.fill_betweenx(support, density, color=fill, alpha=0.62, linewidth=0)
                    marginal.plot(density, support, color=stroke, linewidth=1.9, linestyle=linestyle)

            marginal.set_facecolor("white")
            if dim == 0:
                marginal.set_ylim(bottom=0)
                marginal.spines[["top", "right", "left"]].set_visible(False)
                marginal.spines["bottom"].set_color(SPINE_COLOR)
                marginal.spines["bottom"].set_linewidth(1.15)
                marginal.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            else:
                marginal.set_xlim(left=0)
                marginal.spines[["top", "right", "bottom"]].set_visible(False)
                marginal.spines["left"].set_color(SPINE_COLOR)
                marginal.spines["left"].set_linewidth(1.15)
                marginal.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        return _save_figure(
            fig,
            Path(output_dir) / "vis_pca.png",
            extra_handles=handles,
            legend_ax=ax,
            legend_loc="best",
            grid_axes={ax: "both"},
            skip_style_axes={top, right},
        )


def save_tsne_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
        features = np.concatenate([real_features, fake_features], axis=0)
        perplexity = min(40, max(2, (features.shape[0] - 1) // 3))

        tsne_kwargs = {
            "n_components": 2,
            "verbose": 0,
            "perplexity": perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "random_state": seed,
        }
        iteration_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        tsne_kwargs[iteration_arg] = 300
        points = TSNE(**tsne_kwargs).fit_transform(features)

        sample_count = real_features.shape[0]
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE, layout="constrained")
        handles = _draw_interleaved_embedding(ax, points[:sample_count], points[sample_count:], seed)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        return _save_figure(fig, Path(output_dir) / "vis_tsne.png", extra_handles=handles)


def _sample_values(values, max_points: int = 200000, seed: int = 123) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        flat = flat[rng.choice(flat.shape[0], size=max_points, replace=False)]
    return flat


def _plot_density_on_axis(
    ax: plt.Axes,
    real_values,
    fake_values,
    *,
    xlabel: str,
    seed: int = 123,
    non_negative: bool = False,
) -> None:
    real_v = _sample_values(real_values, seed=seed)
    fake_v = _sample_values(fake_values, seed=seed + 1)
    lo = float(min(real_v.min(), fake_v.min()))
    hi = float(max(real_v.max(), fake_v.max()))
    span = hi - lo
    if span < 1e-6:
        lo -= 1e-3
        hi += 1e-3
    else:
        pad = 0.04 * span
        lo = max(0.0, lo - pad) if non_negative else lo - pad
        hi += pad

    x_grid = np.linspace(lo, hi, 512)
    real_density = _kde_curve(real_v, x_grid)
    fake_density = _kde_curve(fake_v, x_grid)
    peak = float(max(real_density.max(), fake_density.max(), 1e-8))

    ax.fill_between(x_grid, real_density, color=REAL_FILL, alpha=0.58, linewidth=0)
    ax.fill_between(x_grid, fake_density, color=GENERATED_FILL, alpha=0.55, linewidth=0)
    ax.plot(x_grid, real_density, linewidth=CURVE_WIDTH, label="Real", color=REAL_COLOR)
    ax.plot(
        x_grid,
        fake_density,
        linewidth=CURVE_WIDTH,
        linestyle=GENERATED_LINESTYLE,
        label="QuaSiD",
        color=GENERATED_COLOR,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, peak * 1.12)


def _density_figure_from_values(
    real_values,
    fake_values,
    xlabel: str,
    seed: int = 123,
    *,
    non_negative: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, layout="constrained")
    _plot_density_on_axis(ax, real_values, fake_values, xlabel=xlabel, seed=seed, non_negative=non_negative)
    return fig


def save_value_density_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        fig = _density_figure_from_values(
            real_sub.reshape(-1),
            fake_sub.reshape(-1),
            xlabel="Signal value",
            seed=seed,
        )
        return _save_figure(fig, Path(output_dir) / "vis_value_density.png")


def save_diff_density_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        fig = _density_figure_from_values(
            np.diff(real_sub, axis=1).reshape(-1),
            np.diff(fake_sub, axis=1).reshape(-1),
            xlabel=r"Temporal increment $\Delta x_t$",
            seed=seed,
        )
        return _save_figure(fig, Path(output_dir) / "vis_diff_density.png")


def save_spectrum_density_plot(
    real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        real_amp = np.abs(np.fft.rfft(real_sub, axis=1))[:, 1:, :]
        fake_amp = np.abs(np.fft.rfft(fake_sub, axis=1))[:, 1:, :]
        fig = _density_figure_from_values(
            np.log1p(real_amp).reshape(-1),
            np.log1p(fake_amp).reshape(-1),
            xlabel=r"$\log(1 + |\hat{x}(f)|)$",
            seed=seed,
            non_negative=True,
        )
        return _save_figure(fig, Path(output_dir) / "vis_spectrum_density.png")


def save_density_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        fig = _density_figure_from_values(
            real_sub.reshape(-1),
            fake_sub.reshape(-1),
            xlabel="Signal value",
            seed=seed,
        )
        return _save_figure(fig, Path(output_dir) / "vis_density.png")


def _apply_panel_headroom(ax: plt.Axes, values: np.ndarray, *, extra_top: float = 0.10) -> None:
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1e-4)
    ax.set_ylim(lo - 0.09 * span, hi + extra_top * span)


def save_sample_plot(real_data, fake_data, output_dir, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare=4, seed=seed)
        sample_count = real_sub.shape[0]
        seq_len = real_sub.shape[1]
        rows = 2 if sample_count > 2 else 1
        cols = 2 if sample_count > 1 else 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(PANEL_CELL_SIZE[0] * cols, PANEL_CELL_SIZE[1] * rows),
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        flat_axes = np.atleast_1d(axes).flatten()
        steps = np.arange(seq_len)

        for idx in range(sample_count):
            ax = flat_axes[idx]
            row_idx, col_idx = divmod(idx, cols)
            real_series = real_sub[idx, :, 0]
            fake_series = fake_sub[idx, :, 0]
            ax.plot(steps, real_series, label="Real", color=REAL_COLOR, linewidth=1.95)
            ax.plot(
                steps,
                fake_series,
                label="QuaSiD",
                color=GENERATED_COLOR,
                linewidth=1.95,
                linestyle=GENERATED_LINESTYLE,
            )
            ax.set_xlim(0, max(1, seq_len - 1))
            panel_values = np.concatenate([real_series, fake_series])
            _apply_panel_headroom(ax, panel_values, extra_top=0.22 if idx == 0 else 0.10)
            if row_idx == rows - 1 or idx + cols >= sample_count:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("Time step")
            if col_idx == 0:
                ax.set_ylabel("Value (Ch. 1)")

        for ax in flat_axes[sample_count:]:
            ax.axis("off")

        return _save_figure(
            fig,
            Path(output_dir) / "vis_sample.png",
            legend_ax=flat_axes[0],
            legend_loc="upper right",
            legend_ncol=2,
        )


def save_sample_comparison(real_data, fake_data, output_dir, seed: int = 123) -> str:
    return save_sample_plot(real_data, fake_data, output_dir, seed=seed)


def save_temporal_statistics(
    real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_features, fake_features = _sequence_features(real_data, fake_data, compare, seed)
        real_mean = real_features.mean(axis=0)
        fake_mean = fake_features.mean(axis=0)
        real_std = real_features.std(axis=0)
        fake_std = fake_features.std(axis=0)
        steps = np.arange(real_mean.shape[0])

        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, layout="constrained")
        ax.fill_between(
            steps, real_mean - real_std, real_mean + real_std,
            color=REAL_FILL, alpha=0.58, linewidth=0,
        )
        ax.plot(steps, real_mean - real_std, color=REAL_EDGE, linewidth=ENVELOPE_EDGE_WIDTH)
        ax.plot(steps, real_mean + real_std, color=REAL_EDGE, linewidth=ENVELOPE_EDGE_WIDTH)

        ax.fill_between(
            steps, fake_mean - fake_std, fake_mean + fake_std,
            color=GENERATED_FILL, alpha=0.52, linewidth=0,
        )
        ax.plot(
            steps, fake_mean - fake_std, color=GENERATED_EDGE,
            linewidth=ENVELOPE_EDGE_WIDTH, linestyle=GENERATED_LINESTYLE,
        )
        ax.plot(
            steps, fake_mean + fake_std, color=GENERATED_EDGE,
            linewidth=ENVELOPE_EDGE_WIDTH, linestyle=GENERATED_LINESTYLE,
        )

        ax.plot(steps, real_mean, color=REAL_COLOR, linewidth=CURVE_WIDTH)
        ax.plot(
            steps, fake_mean, color=GENERATED_COLOR,
            linewidth=CURVE_WIDTH, linestyle=GENERATED_LINESTYLE,
        )

        lo = float(min((real_mean - real_std).min(), (fake_mean - fake_std).min()))
        hi = float(max((real_mean + real_std).max(), (fake_mean + fake_std).max()))
        span = max(hi - lo, 1e-4)
        ax.set_xlim(0, max(1, steps.shape[0] - 1))
        ax.set_ylim(lo - 0.10 * span, hi + 0.30 * span)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Channel-averaged value")

        handles = {
            "Real mean": Line2D([0], [0], color=REAL_COLOR, linewidth=CURVE_WIDTH),
            r"Real $\pm 1\sigma$": Patch(
                facecolor=REAL_FILL, alpha=0.75, edgecolor=REAL_EDGE, linewidth=ENVELOPE_EDGE_WIDTH
            ),
            "QuaSiD mean": Line2D(
                [0], [0], color=GENERATED_COLOR, linewidth=CURVE_WIDTH, linestyle=GENERATED_LINESTYLE
            ),
            r"QuaSiD $\pm 1\sigma$": Patch(
                facecolor=GENERATED_FILL, alpha=0.75, edgecolor=GENERATED_EDGE, linewidth=ENVELOPE_EDGE_WIDTH
            ),
        }
        return _save_figure(
            fig,
            Path(output_dir) / "vis_temporal_stats.png",
            extra_handles=handles,
            legend_loc="upper right",
            legend_ncol=2,
        )


def _autocorrelation_summary(data: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _as_time_series(data)
    centered = arr - arr.mean(axis=1, keepdims=True)
    scaled = centered / (centered.std(axis=1, keepdims=True) + 1e-6)
    per_sample_acf = np.empty((arr.shape[0], max_lag + 1), dtype=np.float32)
    per_sample_acf[:, 0] = 1.0
    for lag in range(1, max_lag + 1):
        prod = scaled[:, : scaled.shape[1] - lag, :] * scaled[:, lag:, :]
        per_sample_acf[:, lag] = prod.mean(axis=(1, 2))
    mean_acf = per_sample_acf.mean(axis=0)
    q25, q75 = np.quantile(per_sample_acf, [0.25, 0.75], axis=0)
    return mean_acf, q25, q75


def _iqr_legend_handles() -> dict[str, object]:
    return {
        "Real mean": Line2D([0], [0], color=REAL_COLOR, linewidth=CURVE_WIDTH),
        "Real IQR": Patch(facecolor=REAL_FILL, alpha=0.75, edgecolor=REAL_EDGE, linewidth=0.9),
        "QuaSiD mean": Line2D(
            [0], [0], color=GENERATED_COLOR, linewidth=CURVE_WIDTH, linestyle=GENERATED_LINESTYLE
        ),
        "QuaSiD IQR": Patch(
            facecolor=GENERATED_FILL, alpha=0.75, edgecolor=GENERATED_EDGE, linestyle="--", linewidth=0.9
        ),
    }


def _plot_acf_on_axis(ax: plt.Axes, real_sub: np.ndarray, fake_sub: np.ndarray) -> None:
    max_lag = min(128, real_sub.shape[1] - 1)
    lags = np.arange(max_lag + 1)
    real_mean, real_q25, real_q75 = _autocorrelation_summary(real_sub, max_lag)
    fake_mean, fake_q25, fake_q75 = _autocorrelation_summary(fake_sub, max_lag)

    ax.axhline(0.0, color=SPINE_COLOR, linewidth=1.0, linestyle="--", alpha=0.65)
    ax.fill_between(lags, real_q25, real_q75, color=REAL_FILL, alpha=0.58, linewidth=0)
    ax.plot(lags, real_q25, color=REAL_EDGE, linewidth=0.85, alpha=0.75)
    ax.plot(lags, real_q75, color=REAL_EDGE, linewidth=0.85, alpha=0.75)

    ax.fill_between(lags, fake_q25, fake_q75, color=GENERATED_FILL, alpha=0.52, linewidth=0)
    ax.plot(lags, fake_q25, color=GENERATED_EDGE, linewidth=0.85, linestyle=GENERATED_LINESTYLE, alpha=0.75)
    ax.plot(lags, fake_q75, color=GENERATED_EDGE, linewidth=0.85, linestyle=GENERATED_LINESTYLE, alpha=0.75)

    ax.plot(lags, real_mean, label="Real", color=REAL_COLOR, linewidth=CURVE_WIDTH)
    ax.plot(
        lags,
        fake_mean,
        label="QuaSiD",
        color=GENERATED_COLOR,
        linewidth=CURVE_WIDTH,
        linestyle=GENERATED_LINESTYLE,
    )
    ax.set_xlim(0, max(1, max_lag))
    ax.margins(y=0.08)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")


def save_autocorrelation_plot(
    real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, layout="constrained")
        _plot_acf_on_axis(ax, real_sub, fake_sub)
        return _save_figure(
            fig,
            Path(output_dir) / "vis_autocorrelation.png",
            extra_handles=_iqr_legend_handles(),
            legend_loc="best",
            legend_ncol=2,
        )


def _plot_psd_on_axis(ax: plt.Axes, real_sub: np.ndarray, fake_sub: np.ndarray) -> None:
    seq_len = real_sub.shape[1]
    freqs = np.fft.rfftfreq(seq_len, d=1.0)
    real_power = (np.abs(np.fft.rfft(real_sub, axis=1, norm="ortho")) ** 2).mean(axis=2)
    fake_power = (np.abs(np.fft.rfft(fake_sub, axis=1, norm="ortho")) ** 2).mean(axis=2)

    real_log = np.log10(np.maximum(real_power, 1e-8))
    fake_log = np.log10(np.maximum(fake_power, 1e-8))
    r_mean = real_log.mean(axis=0)
    f_mean = fake_log.mean(axis=0)
    r_q25, r_q75 = np.quantile(real_log, [0.25, 0.75], axis=0)
    f_q25, f_q75 = np.quantile(fake_log, [0.25, 0.75], axis=0)

    ax.fill_between(freqs, r_q25, r_q75, color=REAL_FILL, alpha=0.58, linewidth=0)
    ax.plot(freqs, r_q25, color=REAL_EDGE, linewidth=0.85, alpha=0.75)
    ax.plot(freqs, r_q75, color=REAL_EDGE, linewidth=0.85, alpha=0.75)

    ax.fill_between(freqs, f_q25, f_q75, color=GENERATED_FILL, alpha=0.52, linewidth=0)
    ax.plot(freqs, f_q25, color=GENERATED_EDGE, linewidth=0.85, linestyle=GENERATED_LINESTYLE, alpha=0.75)
    ax.plot(freqs, f_q75, color=GENERATED_EDGE, linewidth=0.85, linestyle=GENERATED_LINESTYLE, alpha=0.75)

    ax.plot(freqs, r_mean, label="Real", color=REAL_COLOR, linewidth=CURVE_WIDTH)
    ax.plot(
        freqs,
        f_mean,
        label="QuaSiD",
        color=GENERATED_COLOR,
        linewidth=CURVE_WIDTH,
        linestyle=GENERATED_LINESTYLE,
    )
    ax.set_xlim(float(freqs[0]), float(freqs[-1]) if freqs.size > 1 else 0.5)
    ax.margins(y=0.08)
    ax.set_xlabel("Normalized frequency (cycles/step)")
    ax.set_ylabel(r"$\log_{10} S(f)$")


def save_psd_plot(real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE, layout="constrained")
        _plot_psd_on_axis(ax, real_sub, fake_sub)
        return _save_figure(
            fig,
            Path(output_dir) / "vis_psd.png",
            extra_handles=_iqr_legend_handles(),
            legend_loc="upper right",
            legend_ncol=2,
        )


def save_distribution_overview_plot(
    real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        real_sub, fake_sub = _paired_subset(real_data, fake_data, compare, seed)
        real_features, fake_features = real_sub.mean(axis=2), fake_sub.mean(axis=2)
        pca = PCA(n_components=2).fit(real_features)
        real_pts = pca.transform(real_features)
        fake_pts = pca.transform(fake_features)

        fig, axes = plt.subplots(2, 2, figsize=OVERVIEW_FIGSIZE, layout="constrained")
        ax_pca, ax_val, ax_acf, ax_psd = axes.flatten()

        handles = _draw_interleaved_embedding(ax_pca, real_pts, fake_pts, seed)
        ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
        ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} var.)")

        _plot_density_on_axis(
            ax_val,
            real_sub.reshape(-1),
            fake_sub.reshape(-1),
            xlabel="Signal value",
            seed=seed,
        )
        _plot_acf_on_axis(ax_acf, real_sub, fake_sub)
        _plot_psd_on_axis(ax_psd, real_sub, fake_sub)

        return _save_figure(
            fig,
            Path(output_dir) / "vis_overview.png",
            extra_handles=handles,
            legend_ax=ax_pca,
            legend_loc="best",
        )


def _as_forecast_ensemble(sample_ensemble, real_data) -> tuple[np.ndarray, np.ndarray]:
    real_arr = _as_time_series(real_data)
    ens = np.asarray(sample_ensemble, dtype=np.float32)
    if ens.ndim == 3:
        ens = ens[np.newaxis, ...]
    if ens.ndim != 4:
        raise ValueError(f"Expected forecast ensemble shape (R, N, L, C), got {ens.shape}")
    if ens.shape[1:] != real_arr.shape:
        raise ValueError(f"Forecast ensemble target shape {ens.shape[1:]} != real shape {real_arr.shape}")
    return ens, real_arr


def save_forecast_interval_plot(
    real_data,
    sample_ensemble,
    output_dir,
    contexts=None,
    compare: int = 4,
    seed: int = 123,
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        ens, real_arr = _as_forecast_ensemble(sample_ensemble, real_data)
        ctx = None if contexts is None else _as_time_series(contexts)
        sample_count = min(compare, real_arr.shape[0])
        indices = np.random.default_rng(seed).permutation(real_arr.shape[0])[:sample_count]

        rows = 2 if sample_count > 2 else 1
        cols = 2 if sample_count > 1 else 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(PANEL_CELL_SIZE[0] * cols, PANEL_CELL_SIZE[1] * rows),
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        flat_axes = np.atleast_1d(axes).flatten()
        pred_len = real_arr.shape[1]

        for panel_idx, sample_idx in enumerate(indices):
            ax = flat_axes[panel_idx]
            row_idx, col_idx = divmod(panel_idx, cols)
            forecast_runs = ens[:, sample_idx, :, 0]
            real_future = real_arr[sample_idx, :, 0]
            mean_future = forecast_runs.mean(axis=0)
            q05, q25, q75, q95 = np.quantile(forecast_runs, [0.05, 0.25, 0.75, 0.95], axis=0)

            if ctx is not None:
                history = ctx[sample_idx, :, 0]
                hist_len = len(history)
                history_x = np.arange(-hist_len + 1, 1)
                origin_val = float(history[-1])
                future_x = np.arange(0, pred_len + 1)

                real_curve = np.concatenate([[origin_val], real_future])
                mean_curve = np.concatenate([[origin_val], mean_future])
                q05_curve = np.concatenate([[origin_val], q05])
                q25_curve = np.concatenate([[origin_val], q25])
                q75_curve = np.concatenate([[origin_val], q75])
                q95_curve = np.concatenate([[origin_val], q95])

                ax.axvspan(0, pred_len, color=HORIZON_BG, zorder=0)
                ax.axvline(0, color=SPINE_COLOR, linewidth=1.15, linestyle="--", alpha=0.8, zorder=1)
                ax.plot(history_x, history, color=HISTORY_COLOR, linewidth=1.9, zorder=3)
                panel_values = np.concatenate([history, real_curve, q05_curve, q95_curve])
                x_lo, x_hi = int(history_x[0]), pred_len
            else:
                future_x = np.arange(1, pred_len + 1)
                real_curve, mean_curve = real_future, mean_future
                q05_curve, q25_curve, q75_curve, q95_curve = q05, q25, q75, q95
                panel_values = np.concatenate([real_curve, q05_curve, q95_curve])
                x_lo, x_hi = 1, pred_len

            ax.fill_between(
                future_x, q05_curve, q95_curve, color=GENERATED_FILL, alpha=0.60, linewidth=0, zorder=1
            )
            ax.plot(
                future_x, q05_curve, color=GENERATED_EDGE,
                linewidth=ENVELOPE_EDGE_WIDTH, linestyle="--", zorder=2,
            )
            ax.plot(
                future_x, q95_curve, color=GENERATED_EDGE,
                linewidth=ENVELOPE_EDGE_WIDTH, linestyle="--", zorder=2,
            )
            ax.fill_between(
                future_x, q25_curve, q75_curve, color=CI50_FILL, alpha=0.72, linewidth=0, zorder=2
            )
            ax.plot(future_x, real_curve, color=REAL_COLOR, linewidth=2.0, zorder=4)
            ax.plot(
                future_x, mean_curve, color=GENERATED_COLOR,
                linewidth=2.05, linestyle=GENERATED_LINESTYLE, zorder=5,
            )

            ax.set_xlim(x_lo, x_hi)
            _apply_panel_headroom(ax, panel_values, extra_top=0.30 if panel_idx == 0 else 0.10)
            if row_idx == rows - 1 or panel_idx + cols >= sample_count:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("Time step relative to forecast origin")
            if col_idx == 0:
                ax.set_ylabel("Value (Ch. 1)")

        for ax in flat_axes[sample_count:]:
            ax.axis("off")

        legend_handles: dict[str, object] = {}
        if ctx is not None:
            legend_handles["History"] = Line2D([0], [0], color=HISTORY_COLOR, linewidth=1.95)
        legend_handles["Real"] = Line2D([0], [0], color=REAL_COLOR, linewidth=2.05)
        legend_handles["QuaSiD mean"] = Line2D(
            [0], [0], color=GENERATED_COLOR, linewidth=2.05, linestyle=GENERATED_LINESTYLE
        )
        legend_handles["50% CI"] = Patch(facecolor=CI50_FILL, alpha=0.85, edgecolor=GENERATED_EDGE, linewidth=0.7)
        legend_handles["90% CI"] = Patch(
            facecolor=GENERATED_FILL, alpha=0.75, edgecolor=GENERATED_EDGE, linestyle="--", linewidth=0.9
        )
        return _save_figure(
            fig,
            Path(output_dir) / "forecast_interval.png",
            extra_handles=legend_handles,
            legend_ax=flat_axes[0],
            legend_loc="upper left",
            legend_ncol=3 if ctx is not None else 2,
        )


def save_forecast_samples_plot(
    real_data,
    sample_ensemble,
    output_dir,
    contexts=None,
    max_paths: int = 12,
    compare: int = 4,
    seed: int = 123,
) -> str:
    with matplotlib.rc_context(RC_STYLE):
        ens, real_arr = _as_forecast_ensemble(sample_ensemble, real_data)
        ctx = None if contexts is None else _as_time_series(contexts)
        sample_count = min(compare, real_arr.shape[0])
        rng = np.random.default_rng(seed)
        indices = rng.permutation(real_arr.shape[0])[:sample_count]
        path_count = min(max_paths, ens.shape[0])
        path_indices = rng.permutation(ens.shape[0])[:path_count]

        rows = 2 if sample_count > 2 else 1
        cols = 2 if sample_count > 1 else 1
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(PANEL_CELL_SIZE[0] * cols, PANEL_CELL_SIZE[1] * rows),
            sharex=True,
            sharey=False,
            layout="constrained",
        )
        flat_axes = np.atleast_1d(axes).flatten()
        pred_len = real_arr.shape[1]

        for panel_idx, sample_idx in enumerate(indices):
            ax = flat_axes[panel_idx]
            row_idx, col_idx = divmod(panel_idx, cols)
            real_future = real_arr[sample_idx, :, 0]
            sampled_paths = ens[path_indices, sample_idx, :, 0]
            median_future = np.median(ens[:, sample_idx, :, 0], axis=0)

            if ctx is not None:
                history = ctx[sample_idx, :, 0]
                hist_len = len(history)
                history_x = np.arange(-hist_len + 1, 1)
                origin_val = float(history[-1])
                future_x = np.arange(0, pred_len + 1)
                real_curve = np.concatenate([[origin_val], real_future])
                median_curve = np.concatenate([[origin_val], median_future])

                ax.axvspan(0, pred_len, color=HORIZON_BG, zorder=0)
                ax.axvline(0, color=SPINE_COLOR, linewidth=1.15, linestyle="--", alpha=0.8, zorder=1)
                ax.plot(history_x, history, color=HISTORY_COLOR, linewidth=1.9, zorder=3)
                for path_vals in sampled_paths:
                    path_curve = np.concatenate([[origin_val], path_vals])
                    ax.plot(future_x, path_curve, color=GENERATED_EDGE, alpha=0.55, linewidth=1.05, zorder=2)
                panel_values = np.concatenate([history, real_curve, sampled_paths.reshape(-1)])
                x_lo, x_hi = int(history_x[0]), pred_len
            else:
                future_x = np.arange(1, pred_len + 1)
                real_curve, median_curve = real_future, median_future
                for path_vals in sampled_paths:
                    ax.plot(future_x, path_vals, color=GENERATED_EDGE, alpha=0.55, linewidth=1.05, zorder=2)
                panel_values = np.concatenate([real_curve, sampled_paths.reshape(-1)])
                x_lo, x_hi = 1, pred_len

            ax.plot(future_x, real_curve, color=REAL_COLOR, linewidth=2.0, zorder=4)
            ax.plot(
                future_x, median_curve, color=GENERATED_COLOR,
                linewidth=2.05, linestyle=GENERATED_LINESTYLE, zorder=5,
            )

            ax.set_xlim(x_lo, x_hi)
            _apply_panel_headroom(ax, panel_values, extra_top=0.30 if panel_idx == 0 else 0.10)
            if row_idx == rows - 1 or panel_idx + cols >= sample_count:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel("Time step relative to forecast origin")
            if col_idx == 0:
                ax.set_ylabel("Value (Ch. 1)")

        for ax in flat_axes[sample_count:]:
            ax.axis("off")

        legend_handles: dict[str, object] = {}
        if ctx is not None:
            legend_handles["History"] = Line2D([0], [0], color=HISTORY_COLOR, linewidth=1.95)
        legend_handles["Real"] = Line2D([0], [0], color=REAL_COLOR, linewidth=2.05)
        legend_handles["QuaSiD median"] = Line2D(
            [0], [0], color=GENERATED_COLOR, linewidth=2.05, linestyle=GENERATED_LINESTYLE
        )
        legend_handles["Sample paths"] = Line2D([0], [0], color=GENERATED_EDGE, linewidth=1.35)
        return _save_figure(
            fig,
            Path(output_dir) / "forecast_samples.png",
            extra_handles=legend_handles,
            legend_ax=flat_axes[0],
            legend_loc="upper left",
            legend_ncol=2,
        )


def save_forecast_density_plot(real_data, sample_ensemble, output_dir, seed: int = 123) -> str:
    with matplotlib.rc_context(RC_STYLE):
        ens, real_arr = _as_forecast_ensemble(sample_ensemble, real_data)
        fig = _density_figure_from_values(
            real_arr.reshape(-1),
            ens.reshape(-1),
            xlabel="Forecast value",
            seed=seed,
        )
        return _save_figure(fig, Path(output_dir) / "forecast_density.png")


def save_forecasting_visualizations(
    real_data, sample_ensemble, output_dir, contexts=None, compare: int = 4, seed: int = 123
) -> dict[str, str]:
    return {
        "forecast_interval": save_forecast_interval_plot(
            real_data, sample_ensemble, output_dir, contexts=contexts, compare=compare, seed=seed
        ),
        "forecast_samples": save_forecast_samples_plot(
            real_data, sample_ensemble, output_dir, contexts=contexts, compare=compare, seed=seed
        ),
        "forecast_density": save_forecast_density_plot(real_data, sample_ensemble, output_dir, seed=seed),
    }


def save_distribution_visualizations(
    real_data, fake_data, output_dir, compare: int = 3000, seed: int = 123
) -> dict[str, str]:
    return {
        "overview": save_distribution_overview_plot(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "pca": save_pca_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "tsne": save_tsne_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "density": save_density_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "value_density": save_value_density_plot(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "diff_density": save_diff_density_plot(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "spectrum_density": save_spectrum_density_plot(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "psd": save_psd_plot(real_data, fake_data, output_dir, compare=compare, seed=seed),
        "autocorrelation": save_autocorrelation_plot(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "temporal_stats": save_temporal_statistics(
            real_data, fake_data, output_dir, compare=compare, seed=seed
        ),
        "sample": save_sample_plot(real_data, fake_data, output_dir, seed=seed),
    }
