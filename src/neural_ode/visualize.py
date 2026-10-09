from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from cell_simulator.visualize import COLORS, set_style
from neural_ode.data import TrajectoryData
from neural_ode.evaluate import Prediction
from neural_ode.evaluate_orbit import (
    LatentDiagnostics,
    OrbitFrame,
    fit_phase_per_omega,
    initial_phases,
    orbit_coordinates,
)
from neural_ode.train import TrainHistory

__all__ = [
    "set_style",
    "plot_training_history",
    "plot_trajectory_fit",
    "plot_latent_orbits",
    "plot_parameter_recovery",
    "plot_encoder_grid",
]

OMEGA_CMAP = "viridis"
PHASE_TICKS = (np.arange(0, 2 * np.pi + 0.1, np.pi / 2), ["0", "π/2", "π", "3π/2", "2π"])


def _omega_colors(omegas: np.ndarray, lim: tuple[float, float]) -> np.ndarray:
    return plt.get_cmap(OMEGA_CMAP)(np.clip((omegas - lim[0]) / (lim[1] - lim[0]), 0, 1))


def _omega_colorbar(fig: plt.Figure, axes, lim: tuple[float, float]) -> None:
    sm = plt.cm.ScalarMappable(cmap=OMEGA_CMAP, norm=plt.Normalize(*lim))
    fig.colorbar(sm, ax=axes, label="true ω [rad/s]", shrink=0.8)


def _frame_axes_labels(ax, dims: str = "uvw") -> None:
    labels = {"u": "latent dim 1", "v": "latent dim 2", "w": "ω axis"}
    ax.set_xlabel(labels[dims[0]])
    ax.set_ylabel(labels[dims[1]])
    if len(dims) == 3:
        ax.set_zlabel(labels[dims[2]])


def _uvw(frame: OrbitFrame, latents: np.ndarray) -> np.ndarray:
    """Frame coordinates padded to 3 columns (w = 0 for a 2D latent space)."""
    p = frame.project(latents)
    return p if p.shape[-1] == 3 else np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)


def _is_2d(frame: OrbitFrame) -> bool:
    return frame.basis.shape[1] == 2


def _draw_vector_field(ax: plt.Axes, model, frame: OrbitFrame, uv: np.ndarray, n: int = 30) -> None:
    """Streamlines of the learned drift in the (2D) latent plane, over the extent of ``uv`` (..., 2)."""
    lo, hi = uv.reshape(-1, 2).min(axis=0), uv.reshape(-1, 2).max(axis=0)
    pad = 0.15 * (hi - lo)
    gu, gv = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], n), np.linspace(lo[1] - pad[1], hi[1] + pad[1], n))
    grid = np.column_stack([gu.ravel(), gv.ravel()]) @ frame.basis.T + frame.origin
    with torch.no_grad():
        f = model.drift(torch.zeros(()), torch.as_tensor(grid, dtype=torch.float32)).numpy() @ frame.basis
    ax.streamplot(gu, gv, f[:, 0].reshape(n, n), f[:, 1].reshape(n, n), color="0.8", linewidth=0.6, density=1.2)


def plot_training_history(history: TrainHistory, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Train and validation trajectory MSE against iteration, with the restored (best) iteration marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history.iters, history.train_loss, color=COLORS["best"], label="train")
    ax.plot(history.iters, history.val_loss, color=COLORS["worst"], label="validation")
    ax.axvline(history.best_iter, color=COLORS["observed"], ls=":", lw=1, label="restored")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("trajectory MSE")
    ax.set_title("training history")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_trajectory_fit(data: TrajectoryData, prediction: Prediction, n_show: int = 4) -> plt.Figure:
    """Ground truth vs. reconstruction of a(t) for ``n_show`` trajectories, plus per-frame error over all of them.

    The shown trajectories are the best, the worst and evenly spaced ones in
    between, ranked by their MSE.
    """
    err = np.mean((prediction.observations - data.observations) ** 2, axis=(1, 2))
    order = np.argsort(err)
    shown = order[np.linspace(0, len(order) - 1, n_show).round().astype(int)]

    fig, axes = plt.subplot_mosaic(
        [[str(i) for i in range(n_show)], ["err"] * n_show], figsize=(3.2 * n_show, 5.8), layout="constrained"
    )
    for i, k in enumerate(shown):
        ax = axes[str(i)]
        ax.plot(data.times, data.observations[k, :, 0], color=COLORS["truth"], lw=1.6, label="ground truth")
        ax.plot(data.times, prediction.observations[k, :, 0], color=COLORS["best"], ls="--", lw=1.4, label="Neural ODE")
        ax.set_title(
            f"ω = {data.params['omega'][k]:.2f}, φ = {data.params['phase'][k]:.2f}\nMSE = {err[k]:.1e}", fontsize=9
        )
        ax.set_xlabel("time [s]")
        ax.set_ylim(-0.08, 1.08)
    axes["0"].set_ylabel("a(t)")
    axes["0"].legend(frameon=False, fontsize=8, loc="lower left")

    per_frame = np.mean((prediction.observations - data.observations) ** 2, axis=(0, 2))
    axes["err"].plot(data.times, per_frame, color=COLORS["best"])
    axes["err"].set_yscale("log")
    axes["err"].set_xlabel("time [s]")
    axes["err"].set_ylabel("MSE")
    axes["err"].set_title(f"per-frame MSE, mean over all {len(data)} trajectories")
    return fig


def plot_latent_orbits(
    data: TrajectoryData, prediction: Prediction, diagnostics: LatentDiagnostics, model=None
) -> plt.Figure:
    """The learned family of latent orbits, coloured by true omega, in the :class:`~neural_ode.evaluate.OrbitFrame`.

    2D latent space: the latent plane with streamlines of the learned drift
    (if ``model`` is given); circles start each trajectory.

    3D or higher: (a) 3D view, circles start each trajectory; (b) top view
    onto the rotation plane; (c) side view along the omega axis.
    """
    lim = (float(data.params["omega"].min()), float(data.params["omega"].max()))
    colors = _omega_colors(data.params["omega"], lim)
    uvw = _uvw(diagnostics.frame, prediction.latents)

    if _is_2d(diagnostics.frame):
        fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
        if model is not None:
            _draw_vector_field(ax, model, diagnostics.frame, uvw[..., :2])
        for path, color in zip(uvw, colors):
            ax.plot(path[:, 0], path[:, 1], color=color, lw=0.9)
        ax.scatter(uvw[:, 0, 0], uvw[:, 0, 1], color=colors, s=18, edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_aspect("equal")
        ax.set_title("latent orbits over the learned vector field", fontsize=10)
        _frame_axes_labels(ax, "uv")
        _omega_colorbar(fig, ax, lim)
        return fig

    fig = plt.figure(figsize=(17, 5), layout="constrained")
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    ax_top, ax_side = (fig.add_subplot(1, 3, i) for i in (2, 3))
    for path, color in zip(uvw, colors):
        ax3.plot(*path.T, color=color, lw=0.9)
        ax_top.plot(path[:, 0], path[:, 1], color=color, lw=0.9)
        ax_side.plot(path[:, 0], path[:, 2], color=color, lw=0.9)
    ax3.scatter(*uvw[:, 0].T, color=colors, s=14, edgecolor="black", linewidth=0.4, depthshade=False)
    ax3.set_title("(a) latent orbits", fontsize=10)
    _frame_axes_labels(ax3)
    ax3.view_init(elev=20, azim=-60)
    ax_top.scatter(uvw[:, 0, 0], uvw[:, 0, 1], color=colors, s=18, edgecolor="black", linewidth=0.4, zorder=3)
    ax_top.set_aspect("equal")
    ax_top.set_title("(b) top view: rotation plane")
    _frame_axes_labels(ax_top, "uv")
    ax_side.set_title("(c) side view: stacking along the ω axis")
    _frame_axes_labels(ax_side, "uw")
    _omega_colorbar(fig, [ax3, ax_top, ax_side], lim)
    return fig


def plot_parameter_recovery(
    diagnostics: LatentDiagnostics, probe: TrajectoryData, probe_prediction: Prediction
) -> plt.Figure:
    """Recovered vs. true parameters: the Neural ODE counterpart of the random search.

    (a) Winding frequency of each test trajectory's latent orbit against its
    true omega; the diagonal is perfect recovery. (b) Initial latent phase of
    each ``probe`` trajectory (a :func:`~neural_ode.data.parameter_grid`)
    against its true phase, after removing the best rotation/reflection for
    each omega level, since the data cannot fix an omega-dependent rotation.
    """
    fig, (ax_w, ax_p) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    lim = (float(probe.params["omega"].min()), float(probe.params["omega"].max()))
    true, hat = diagnostics.omega_true, diagnostics.omega_hat
    ax_w.plot([true.min(), true.max()], [true.min(), true.max()], color="0.6", lw=1, label="perfect recovery")
    ax_w.scatter(true, hat, color=COLORS["best"], s=26, edgecolor="black", linewidth=0.4, zorder=3)
    rel = np.median(np.abs(hat - true) / true)
    ax_w.set(xlabel="true ω [rad/s]", ylabel="recovered ω̂ [rad/s]")
    ax_w.set_title(f"(a) frequency per test trajectory, median error {100 * rel:.1f} %")
    ax_w.legend(frameon=False, fontsize=8)

    coords = orbit_coordinates(probe_prediction.latents, diagnostics.frame)
    latent = initial_phases(coords(probe_prediction.latents))
    fits = fit_phase_per_omega(latent, probe.params["phase"], probe.params["omega"])
    aligned = np.empty_like(latent)
    for w, fit in fits.items():
        sel = probe.params["omega"] == w
        aligned[sel] = np.mod(fit.orientation * (latent[sel] - fit.offset), 2 * np.pi)
    rms = np.sqrt(np.mean([f.rms_error**2 for f in fits.values()]))
    ax_p.plot([0, 2 * np.pi], [0, 2 * np.pi], color="0.6", lw=1)
    ax_p.scatter(
        probe.params["phase"],
        aligned,
        color=_omega_colors(probe.params["omega"], lim),
        s=22,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_p.set_xticks(*PHASE_TICKS)
    ax_p.set_yticks(*PHASE_TICKS)
    ax_p.set(xlabel="true initial phase φ [rad]", ylabel="latent phase, aligned per ω [rad]")
    ax_p.set_title(f"(b) phase of unseen trajectories, RMS error {rms:.3f} rad")
    _omega_colorbar(fig, ax_p, lim)
    return fig


def plot_encoder_grid(probe: TrajectoryData, z0: np.ndarray, frame: OrbitFrame, n_phases: int) -> plt.Figure:
    """Where the encoder puts z0 for a regular grid of unseen (omega, phi).

    ``probe`` must come from :func:`~neural_ode.data.parameter_grid` with
    ``n_phases`` phases, and ``z0`` be its encoded initial states. Closed
    loops join equal omega (one ring per frequency); open lines join equal
    phi. Rings stacked along the omega axis are a cylinder, rings nested in
    the rotation plane are the 2D layout, crossing or collapsing lines mean
    the encoder confuses parameters. For a 2D latent space the plane is
    shown directly; otherwise (a) 3D view, (b) top view.
    """
    lim = (float(probe.params["omega"].min()), float(probe.params["omega"].max()))
    grid = _uvw(frame, z0).reshape(-1, n_phases, 3)
    omegas = probe.params["omega"].reshape(-1, n_phases)[:, 0]
    if _is_2d(frame):
        fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
        for ring, color in zip(grid, _omega_colors(omegas, lim)):
            closed = np.vstack([ring, ring[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.4)
        for j in range(n_phases):
            ax.plot(grid[:, j, 0], grid[:, j, 1], color="0.6", lw=0.7, ls="--")
        ax.scatter(grid[:, 0, 0], grid[:, 0, 1], color="black", s=16, zorder=3, label="φ = 0")
        ax.set_aspect("equal")
        ax.set_title("encoded z0: rings = equal ω, dashed = equal φ", fontsize=10)
        _frame_axes_labels(ax, "uv")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _omega_colorbar(fig, ax, lim)
        return fig
    fig = plt.figure(figsize=(12, 5.2), layout="constrained")
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)
    for ring, color in zip(grid, _omega_colors(omegas, lim)):
        closed = np.vstack([ring, ring[:1]])
        ax3.plot(*closed.T, color=color, lw=1.4)
        ax_top.plot(closed[:, 0], closed[:, 1], color=color, lw=1.4)
    for j in range(n_phases):
        ax3.plot(*grid[:, j].T, color="0.6", lw=0.7, ls="--")
        ax_top.plot(grid[:, j, 0], grid[:, j, 1], color="0.6", lw=0.7, ls="--")
    ax3.scatter(*grid[:, 0].T, color="black", s=16, depthshade=False, label="φ = 0")
    ax_top.scatter(grid[:, 0, 0], grid[:, 0, 1], color="black", s=16, zorder=3, label="φ = 0")
    ax3.set_title("(a) encoded z0: rings = equal ω, dashed = equal φ", fontsize=10)
    _frame_axes_labels(ax3)
    ax3.view_init(elev=20, azim=-60)
    ax3.legend(frameon=False, fontsize=8, loc="upper left")
    ax_top.set_aspect("equal")
    ax_top.set_title("(b) top view: rotation plane")
    _frame_axes_labels(ax_top, "uv")
    _omega_colorbar(fig, [ax3, ax_top], lim)
    return fig
