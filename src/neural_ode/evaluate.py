"""Generic evaluation of a trained LatentODE.

Metrics (:func:`evaluate`, a flat dict of floats, ready for ``wandb.log`` or JSON):

- Reconstruction: nMSE on train, val and test, and the val/train ratio
  (the overfitting signal).
- Dynamics: forecast nMSE beyond the training window. The encoder sees the
  first ``context_frames`` frames, the latent ODE is integrated on, and the
  decoded continuation is compared with the data.

Figures (:func:`make_figures`, a dict of named matplotlib figures): the data,
trajectory fits, latent paths, forecasts, and a parameter grid moved by the
learned dynamics.
Latent spaces of more than two dimensions are shown on their first two
principal components (:class:`Projection`).

Nothing here assumes a particular system. The oscillator-specific diagnostics
(orbit frames, winding frequency, phase fits) live in ``neural_ode.evaluate_orbit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_ode.data import TrajectoryData
from neural_ode.model import LatentODE


@dataclass
class Prediction:
    """Model output for every trajectory of a :class:`~neural_ode.data.TrajectoryData`."""

    observations: np.ndarray  # (N, T, obs_dim)
    latents: np.ndarray  # (N, T, latent_dim)


@dataclass
class GridAxis:
    """One parameter of the grid in :func:`plot_latent_flow`."""

    name: str
    values: np.ndarray
    periodic: bool  # an angle: its lines are drawn closed


@dataclass
class Projection:
    """A fixed 2D view of the latent space, shared by figures so that their axes agree.

    Two latent dimensions are shown as they are. More are projected on the first two
    principal components of the latents the projection was fitted on.
    """

    mean: np.ndarray  # (latent_dim,)
    components: np.ndarray  # (2, latent_dim)
    labels: tuple[str, str]

    @classmethod
    def fit(cls, latents: np.ndarray) -> Projection:
        """The view for ``latents`` of shape (..., latent_dim), with latent_dim >= 2."""
        dim = latents.shape[-1]
        if dim == 2:
            return cls(np.zeros(2), np.eye(2), ("$z_1$", "$z_2$"))
        flat = latents.reshape(-1, dim)
        mean = flat.mean(0)
        _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
        share = s**2 / np.sum(s**2)
        return cls(mean, vt[:2], (f"PC 1 ({share[0]:.0%} of variance)", f"PC 2 ({share[1]:.0%})"))

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        """Latents (..., latent_dim) in the 2D view, shape (..., 2)."""
        return (latents - self.mean) @ self.components.T


# --------------------------------------------------------------------------- prediction


@torch.no_grad()
def predict(model: LatentODE, data: TrajectoryData, context_frames: int | None = None) -> Prediction:
    """Encode, integrate and decode every trajectory in ``data``.

    With ``context_frames`` set, the encoder sees only the first ``context_frames``
    frames, and the model is integrated over all of ``data.times``: a forecast.
    """
    model.eval()
    times, obs = data.tensors()
    if context_frames is None:
        pred, latents = model(times, obs)
    else:
        latents = model.integrate(model.encoder(obs[:, :context_frames]), times)
        pred = model.decoder(latents)
    return Prediction(pred.numpy(), latents.numpy())


@torch.no_grad()
def encode(model: LatentODE, data: TrajectoryData) -> np.ndarray:
    """Encoded initial states z0 of every trajectory in ``data``, shape (N, latent_dim)."""
    model.eval()
    _, obs = data.tensors()
    return model.encoder(obs).numpy()


# --------------------------------------------------------------------------- metrics


def nmse(pred: np.ndarray, target: np.ndarray) -> float:
    """MSE divided by the variance of the target: 1 means "no better than the mean", 0 is perfect."""
    return float(np.mean((pred - target) ** 2) / np.var(target))


def evaluate(
    model: LatentODE,
    splits: dict[str, TrajectoryData],
    forecast_data: TrajectoryData | None = None,
    *,
    context_frames: int | None = None,
) -> dict[str, float]:
    """All scalar metrics of one model, as a flat dict.

    Args:
        splits: "train", "val" and "test" data.
        forecast_data: Trajectories longer than the training window, or None to skip forecasting.
        context_frames: Frames the encoder sees when forecasting (the training window length).
    """
    metrics = {}
    for name in ("train", "val", "test"):
        data = splits[name]
        metrics[f"{name}/nmse"] = nmse(predict(model, data).observations, data.observations)
    metrics["gap/val_over_train"] = metrics["val/nmse"] / metrics["train/nmse"]

    if forecast_data is not None:
        if not context_frames:
            raise ValueError("forecasting needs context_frames")
        pred = predict(model, forecast_data, context_frames).observations
        obs = forecast_data.observations
        metrics["forecast/nmse_context"] = nmse(pred[:, :context_frames], obs[:, :context_frames])
        metrics["forecast/nmse_future"] = nmse(pred[:, context_frames:], obs[:, context_frames:])
    return metrics


# --------------------------------------------------------------------------- figures


def make_figures(
    model: LatentODE,
    splits: dict[str, TrajectoryData],
    forecast_data: TrajectoryData | None = None,
    grid_data: TrajectoryData | None = None,
    grid_axes: Sequence[GridAxis] | None = None,
    *,
    context_frames: int | None = None,
    color_param: str | None = None,
    snapshot_times: Sequence[float] = (0.0, 2.5, 5.0, 10.0),
) -> dict[str, plt.Figure]:
    """Standard figures of one model. Forecast and latent-flow figures only when their data are given."""
    test = splits["test"]
    pred = predict(model, test)
    projection = Projection.fit(pred.latents) if pred.latents.shape[-1] >= 2 else None
    figures = {
        "data": plot_data(splits, color_param),
        "fits": plot_fits(test, pred),
        "latents": plot_latents(model, test, pred, color_param, projection=projection),
    }
    if forecast_data is not None and context_frames:
        figures["forecast"] = plot_forecast(
            forecast_data, predict(model, forecast_data, context_frames), context_frames
        )
    if grid_data is not None and grid_axes is not None and projection is not None:
        figures["latent_flow"] = plot_latent_flow(
            grid_data, predict(model, grid_data), grid_axes, snapshot_times, projection
        )
    return figures


def plot_data(splits: dict[str, TrajectoryData], color_param: str | None = None, n_examples: int = 4) -> plt.Figure:
    """Example training trajectories, and the true parameters of each split (if the data have any)."""
    train = splits["train"]
    params = {name: _param_dict(splits[name].params) for name in ("train", "val", "test")}
    names = list(params["train"])
    key = color_param if color_param in params["train"] else (names[0] if names else None)

    if names:
        fig, (ax_p, ax_x) = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained", width_ratios=[1, 1.8])
    else:
        fig, ax_x = plt.subplots(figsize=(7, 3.8), layout="constrained")

    order = np.argsort(params["train"][key]) if key else np.arange(len(train.observations))
    for k in order[np.linspace(0, len(order) - 1, n_examples).round().astype(int)]:
        label = ", ".join(f"{n} = {params['train'][n][k]:.2f}" for n in names) or None
        ax_x.plot(train.times, train.observations[k, :, 0], label=label)
    ax_x.set(xlabel="time [s]", ylabel="observation 0", title="example training trajectories")
    if names:
        ax_x.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

        markers = {"train": "o", "val": "s", "test": "^"}
        if len(names) >= 2:
            other = next(n for n in names if n != key)
            for split, marker in markers.items():
                ax_p.scatter(params[split][key], params[split][other], s=12, marker=marker, alpha=0.8, label=split)
            ax_p.set(xlabel=key, ylabel=other, title="true parameters")
        else:
            for split in markers:
                ax_p.hist(params[split][key], bins=20, alpha=0.6, label=split)
            ax_p.set(xlabel=key, ylabel="count", title="true parameter")
        ax_p.legend(frameon=False, fontsize=8)
    return fig


def plot_fits(data: TrajectoryData, pred: Prediction, channel: int = 0) -> plt.Figure:
    """The best and worst test trajectory by per-trajectory MSE."""
    per_traj = ((pred.observations - data.observations) ** 2).mean(axis=(1, 2))
    order = np.argsort(per_traj)
    picks = {"best": order[0], "worst": order[-1]}
    fig, axes = plt.subplots(1, len(picks), figsize=(4.5 * len(picks), 3.2), layout="constrained", sharey=True)
    for ax, (label, k) in zip(axes, picks.items()):
        ax.plot(data.times, data.observations[k, :, channel], color="0.3", label="data")
        ax.plot(data.times, pred.observations[k, :, channel], ls="--", label="model")
        ax.set(xlabel="time [s]", title=f"{label}: MSE {per_traj[k]:.1e}")
    axes[0].set_ylabel(f"observation {channel}")
    axes[0].legend(frameon=False, fontsize=8)
    return fig


def plot_latents(
    model: LatentODE,
    data: TrajectoryData,
    pred: Prediction,
    color_param: str | None = None,
    n_paths: int | None = 10,
    projection: Projection | None = None,
) -> plt.Figure:
    """Latent paths coloured by a true parameter. For a 2D latent space, over streamlines of the drift.

    Black dots are the encoded starting points z0. The axes are the model's own latent
    coordinates (see :class:`Projection` for more than two dimensions), so the orientation
    differs between trained models. Only ``n_paths`` trajectories are drawn (all if None),
    spread evenly over ``color_param`` when given.
    """
    z = pred.latents
    dim = z.shape[-1]
    values = _param_dict(data.params).get(color_param) if color_param else None
    colors, mappable = _colors(values, len(z))

    idx = np.arange(len(z))
    if n_paths is not None and n_paths < len(z):
        order = np.argsort(values) if values is not None else idx
        idx = order[np.linspace(0, len(z) - 1, n_paths).round().astype(int)]

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    if dim == 1:
        for i in idx:
            ax.plot(data.times, z[i, :, 0], color=colors[i], lw=0.8)
        ax.set(xlabel="time [s]", ylabel="$z_1$", title="latent paths")
    else:
        projection = projection or Projection.fit(z)
        u = projection(z)
        if dim == 2:
            _drift_streamlines(ax, model, z)
        for i in idx:
            ax.plot(u[i, :, 0], u[i, :, 1], color=colors[i], lw=0.8)
        ax.plot(u[idx, 0, 0], u[idx, 0, 1], "k.", ms=3, label="$z_0$")
        title = "latent paths" if dim == 2 else f"latent paths ({dim} dims, principal components)"
        ax.set(xlabel=projection.labels[0], ylabel=projection.labels[1], title=title, aspect="equal")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=f"true {color_param}", shrink=0.8)
    return fig


def plot_forecast(
    data: TrajectoryData,
    pred: Prediction,
    context_frames: int,
    channel: int = 0,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> plt.Figure:
    """Forecast error over time relative to each trajectory's own variance, and example forecasts.

    The squared error of a trajectory is divided by the variance of its observations in the
    context window, so 1 means "no better than that trajectory's mean" whatever its amplitude.
    The band is the 25-75 % range over trajectories. The examples are the trajectories at
    ``quantiles`` of that relative error after the context window, from good to bad.
    """
    scale = data.observations[:, :context_frames].var(axis=(1, 2))  # (N,)
    err = ((pred.observations - data.observations) ** 2).mean(-1) / scale[:, None]  # (N, T)
    future = err[:, context_frames:].mean(1)
    order = np.argsort(future)
    t_end = data.times[context_frames - 1]
    n = 1 + len(quantiles)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), layout="constrained")

    low, median, high = np.quantile(err, [0.25, 0.5, 0.75], axis=0)
    axes[0].fill_between(data.times, low, high, alpha=0.25, label="25-75 %")
    axes[0].semilogy(data.times, median, label="median")
    axes[0].semilogy(data.times, err.mean(0), label="mean")
    axes[0].axhline(1.0, color="0.6", lw=0.8)  # the trajectory's own mean as a forecast
    axes[0].set(xlabel="time [s]", ylabel="squared error / trajectory variance", title="forecast error")
    axes[0].legend(frameon=False, fontsize=8)

    for ax, q in zip(axes[1:], quantiles):
        k = order[int(round(q * (len(order) - 1)))]
        ax.plot(data.times, data.observations[k, :, channel], color="0.3", label="data")
        ax.plot(data.times, pred.observations[k, :, channel], ls="--", label="model")
        ax.set(xlabel="time [s]", title=f"{q:.0%} quantile: relative error {future[k]:.2g}")
    axes[1].set_ylabel(f"observation {channel}")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.axvline(t_end, color="k", lw=0.8, ls=":")  # end of what the encoder sees
    return fig


def plot_latent_flow(
    data: TrajectoryData,
    pred: Prediction,
    axes: Sequence[GridAxis],
    snapshot_times: Sequence[float],
    projection: Projection | None = None,
) -> plt.Figure:
    """A regular grid of unseen parameters, encoded and then moved by the learned dynamics.

    Each panel shows where the grid's trajectories are at one time. Coloured lines join
    equal values of the first grid parameter; dashed lines join equal values of the second.
    Black dots mark the first value of the second parameter on each coloured line
    (phase 0 for an oscillator), so their spread shows how far each line has turned.
    A grid that stays untangled over time means the dynamics keep parameters apart.
    The latents are shown in ``projection`` (fitted on the grid's own latents if None).
    """
    a, b = axes
    params = _param_dict(data.params)
    projection = projection or Projection.fit(pred.latents)
    u = projection(pred.latents)  # (N, T, 2)
    lines = []  # one index array per value of a, ordered along b
    for value in a.values:
        idx = np.where(np.isclose(params[a.name], value))[0]
        lines.append(idx[np.argsort(params[b.name][idx])])

    shown = [int(np.searchsorted(data.times, t)) for t in snapshot_times if t <= data.times[-1] + 1e-9]
    pts = u[:, shown].reshape(-1, 2)
    pad = 0.08 * (pts.max(0) - pts.min(0))
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    cmap, norm = plt.get_cmap("viridis"), mpl.colors.Normalize(a.values.min(), a.values.max())

    fig, panels = plt.subplots(1, len(shown), figsize=(4 * len(shown), 4.3), layout="constrained", squeeze=False)
    for ax, k in zip(panels[0], shown):
        for value, idx in zip(a.values, lines):
            p = u[idx, k]
            if b.periodic:
                p = np.vstack([p, p[:1]])
            ax.plot(p[:, 0], p[:, 1], color=cmap(norm(value)), lw=1.5)
        for j in range(min(len(idx) for idx in lines)):
            p = u[[idx[j] for idx in lines], k]
            ax.plot(p[:, 0], p[:, 1], ls="--", color="0.6", lw=0.7)
        first = u[[idx[0] for idx in lines], k]
        ax.scatter(first[:, 0], first[:, 1], color="k", s=15, zorder=3)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", title=f"t = {data.times[k]:g} s")
    panels[0, 0].set(xlabel=projection.labels[0], ylabel=projection.labels[1])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=panels[0], label=f"true {a.name}", shrink=0.8)
    fig.suptitle(f"Parameter grid moved by the learned dynamics (dots: {b.name} = {b.values[0]:g})")
    return fig


# --------------------------------------------------------------------------- helpers


@torch.no_grad()
def _drift_streamlines(ax: plt.Axes, model: LatentODE, z: np.ndarray, n_grid: int = 30) -> None:
    """Streamlines of the learned drift over the region the latents occupy (2D latent space, autonomous drift)."""
    flat = z.reshape(-1, 2)
    lo, hi = flat.min(0), flat.max(0)
    pad = 0.1 * (hi - lo)
    xx, yy = np.meshgrid(
        np.linspace(lo[0] - pad[0], hi[0] + pad[0], n_grid),
        np.linspace(lo[1] - pad[1], hi[1] + pad[1], n_grid),
    )
    dtype = next(model.parameters()).dtype
    grid = torch.as_tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=dtype)
    f = model.drift(torch.zeros((), dtype=dtype), grid).numpy()
    ax.streamplot(
        xx,
        yy,
        f[:, 0].reshape(xx.shape),
        f[:, 1].reshape(yy.shape),
        color="0.8",
        density=1.0,
        linewidth=0.6,
        arrowsize=0.7,
    )


def _colors(values: np.ndarray | None, n: int) -> tuple[list, mpl.cm.ScalarMappable | None]:
    """One colour per trajectory from ``values`` (viridis), or a single colour if there are none."""
    if values is None:
        return ["C0"] * n, None
    cmap, norm = plt.get_cmap("viridis"), mpl.colors.Normalize(values.min(), values.max())
    return [cmap(norm(v)) for v in values], mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _param_dict(params: Any) -> dict[str, np.ndarray]:
    """Parameters as {name: (N,) array}, whether stored as a dict or a structured array."""
    if params is None:
        return {}
    if hasattr(params, "dtype") and params.dtype.names:
        return {name: np.asarray(params[name], dtype=float) for name in params.dtype.names}
    return {name: np.asarray(values, dtype=float) for name, values in dict(params).items()}
