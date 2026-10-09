"""Diagnostics for the adjoint gradient of :mod:`neural_ode.solvers`.

- Gradient agreement (:func:`compare_gradients`): the adjoint gradient against
  backpropagation through the solver, on the same batch. The adjoint is
  optimise-then-discretise, so the two agree only up to the RK4 error.
- Cost (:func:`profile_cost`): memory kept for the backward pass and wall-clock
  time against the number of solver steps.

:func:`plot_adjoint_summary` shows both in one figure. Everything works on a trained
or untrained :class:`~neural_ode.model.LatentODE` and leaves the model unchanged.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_ode.data import TrajectoryData
from neural_ode.model import LatentODE
from neural_ode.train import trajectory_mse

MODULE_COLORS = {"encoder": "C0", "drift": "C1", "decoder": "C2"}


@contextmanager
def solver_settings(
    model: LatentODE, gradient: str | None = None, steps_per_frame: int | None = None
) -> Iterator[LatentODE]:
    """Temporarily set ``model.gradient`` and ``model.steps_per_frame``; the old values are restored on exit."""
    old = model.gradient, model.steps_per_frame
    if gradient is not None:
        model.gradient = gradient
    if steps_per_frame is not None:
        model.steps_per_frame = steps_per_frame
    try:
        yield model
    finally:
        model.gradient, model.steps_per_frame = old


def batch_tensors(
    data: TrajectoryData, batch_size: int | None = None, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(times, observations)`` of the first ``batch_size`` trajectories (all if None), as ``dtype``."""
    times, obs = data.tensors()
    if batch_size is not None:
        obs = obs[:batch_size]
    return times.to(dtype), obs.to(dtype)


def _loss(model: LatentODE, times: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    pred, _ = model(times, obs)
    return trajectory_mse(pred, obs)


def _gradients(model: LatentODE, times: torch.Tensor, obs: torch.Tensor) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = _loss(model, times, obs)
    loss.backward()
    grads = {n: p.grad.detach().flatten().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), grads


# --------------------------------------------------------------------------- gradient agreement


@dataclass
class GradientComparison:
    """Backprop and adjoint gradients of the same loss, for one solver setting."""

    steps_per_frame: int
    names: list[str]  # parameter tensors, in model order
    rel_error: np.ndarray  # per tensor: |g_adj - g_bp| / |g_bp|
    total_rel_error: float  # over all parameters at once
    backprop: np.ndarray  # all gradients, concatenated
    adjoint: np.ndarray
    module: np.ndarray  # top-level module of each entry of ``backprop``
    loss_backprop: float
    loss_adjoint: float
    missing: list[str]  # tensors with a backprop gradient but no adjoint gradient


def compare_gradients(
    model: LatentODE,
    data: TrajectoryData,
    batch_size: int | None = 32,
    steps_per_frame: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> GradientComparison:
    """Gradients of the training loss on one batch, from backprop and from the adjoint.

    The model is copied and cast to ``dtype``; float64 keeps round-off well below
    the solver error, so the difference measured is the discretisation gap between
    the two methods. The encoder is deterministic, so both passes see the same z0.

    Args:
        model: The model to check; it is not modified.
        data: Trajectories; the first ``batch_size`` form the batch.
        batch_size: Batch size (None: all trajectories).
        steps_per_frame: RK4 sub-steps per frame (None: the model's own setting).
        dtype: Precision of the check.

    Returns:
        The two gradients and their relative errors.
    """
    m = copy.deepcopy(model).to(dtype)
    times, obs = batch_tensors(data, batch_size, dtype)
    with solver_settings(m, "backprop", steps_per_frame):
        loss_bp, g_bp = _gradients(m, times, obs)
    with solver_settings(m, "adjoint", steps_per_frame):
        loss_adj, g_adj = _gradients(m, times, obs)

    names = [n for n in g_bp if n in g_adj]
    rel = np.array([float((g_adj[n] - g_bp[n]).norm() / (g_bp[n].norm() + 1e-300)) for n in names])
    bp = torch.cat([g_bp[n] for n in names]).cpu().numpy()
    adj = torch.cat([g_adj[n] for n in names]).cpu().numpy()
    module = np.concatenate([np.full(g_bp[n].numel(), n.split(".")[0]) for n in names])
    return GradientComparison(
        steps_per_frame=m.steps_per_frame if steps_per_frame is None else steps_per_frame,
        names=names,
        rel_error=rel,
        total_rel_error=float(np.linalg.norm(adj - bp) / np.linalg.norm(bp)),
        backprop=bp,
        adjoint=adj,
        module=module,
        loss_backprop=loss_bp,
        loss_adjoint=loss_adj,
        missing=sorted(set(g_bp) - set(g_adj)),
    )


# --------------------------------------------------------------------------- memory and time


def saved_tensor_bytes(fn, exclude: Sequence[torch.Tensor] = ()) -> tuple[torch.Tensor, int]:
    """Run ``fn()`` and count the bytes autograd keeps for the backward pass.

    Every tensor saved for backward (by built-in ops and by custom
    ``autograd.Function``s through ``save_for_backward``) is recorded once per
    underlying storage, so views and repeated saves are not double counted.
    Storages of ``exclude`` (the parameters) are left out. This is the memory
    that grows with the number of solver steps, and it is measured the same way
    on CPU and GPU.

    Returns:
        ``(fn(), bytes kept for backward)``.
    """
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    storages: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        s = t.untyped_storage()
        if s.data_ptr() not in skip:
            storages[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(storages.values())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_cost(
    model: LatentODE,
    data: TrajectoryData,
    steps_per_frame: Sequence[int] = (1, 2, 4, 8, 16, 32),
    batch_size: int | None = 32,
    n_repeat: int = 5,
) -> list[dict[str, float]]:
    """Memory and time of one training step (forward and backward), for both gradient methods.

    The number of solver steps is varied through ``steps_per_frame``, so data and
    loss stay the same. Memory is :func:`saved_tensor_bytes`; on CUDA the peak
    allocation of the whole step is recorded as well. Times are medians over
    ``n_repeat`` repeats, after one warm-up step.

    Returns:
        One row per (gradient, steps_per_frame): ``gradient``, ``steps_per_frame``, ``n_steps``
        (RK4 steps over the window), ``saved_mb``, ``peak_mb`` (NaN off CUDA), ``forward_s``, ``backward_s``.
    """
    device = next(model.parameters()).device
    times, obs = batch_tensors(data, batch_size, next(model.parameters()).dtype)
    times, obs = times.to(device), obs.to(device)
    params = list(model.parameters())
    rows = []
    for spf in steps_per_frame:
        for gradient in ("backprop", "adjoint"):
            with solver_settings(model, gradient, spf):
                _loss(model, times, obs).backward()
                fwd, bwd, saved, peak = [], [], 0, []
                for _ in range(n_repeat):
                    model.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        base = torch.cuda.memory_allocated(device)
                    _sync(device)
                    t0 = time.perf_counter()
                    loss, saved = saved_tensor_bytes(lambda: _loss(model, times, obs), exclude=params)
                    _sync(device)
                    t1 = time.perf_counter()
                    loss.backward()
                    _sync(device)
                    t2 = time.perf_counter()
                    fwd.append(t1 - t0)
                    bwd.append(t2 - t1)
                    if device.type == "cuda":
                        peak.append((torch.cuda.max_memory_allocated(device) - base) / 2**20)
                    del loss
                model.zero_grad(set_to_none=True)
            rows.append(
                {
                    "gradient": gradient,
                    "steps_per_frame": spf,
                    "n_steps": spf * (len(times) - 1),
                    "saved_mb": saved / 2**20,
                    "peak_mb": float(np.median(peak)) if peak else float("nan"),
                    "forward_s": float(np.median(fwd)),
                    "backward_s": float(np.median(bwd)),
                }
            )
    return rows


def plot_adjoint_summary(comparison: GradientComparison, rows: Sequence[dict[str, float]]) -> plt.Figure:
    """(a) adjoint against backprop gradient, (b) memory and (c) time per training step against the number of RK4 steps.

    Args:
        comparison: Result of :func:`compare_gradients`.
        rows: Result of :func:`profile_cost`.
    """
    fig, (ax_g, ax_m, ax_t) = plt.subplots(
        1, 3, figsize=(15.5, 4.4), layout="constrained", width_ratios=[1, 1.15, 1.15]
    )
    for name, color in MODULE_COLORS.items():
        sel = comparison.module == name
        if sel.any():
            ax_g.scatter(comparison.backprop[sel], comparison.adjoint[sel], s=5, alpha=0.6, color=color, label=name)
    lim = 1.05 * np.abs(np.concatenate([comparison.backprop, comparison.adjoint])).max()
    ax_g.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax_g.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        aspect="equal",
        xlabel="backprop through RK4",
        ylabel="adjoint",
        title=f"(a) every gradient entry, rel. error {comparison.total_rel_error:.1e}",
    )
    ax_g.legend(frameon=False, fontsize=8, markerscale=2)

    for gradient, color in (("backprop", "C3"), ("adjoint", "C0")):
        r = [x for x in rows if x["gradient"] == gradient]
        n = [x["n_steps"] for x in r]
        ax_m.loglog(n, [x["saved_mb"] for x in r], "o-", color=color, label=gradient)
        if not np.isnan(r[0]["peak_mb"]):
            ax_m.loglog(n, [x["peak_mb"] for x in r], "s:", color=color, ms=3, label=f"{gradient}, CUDA peak")
        ax_t.loglog(n, [x["forward_s"] + x["backward_s"] for x in r], "o-", color=color, label=f"{gradient}, total")
        ax_t.loglog(n, [x["backward_s"] for x in r], ".--", color=color, lw=0.9, label=f"{gradient}, backward only")
    ax_m.set(xlabel="RK4 steps over the window", ylabel="memory kept for backward [MB]", title="(b) memory")
    ax_t.set(xlabel="RK4 steps over the window", ylabel="time per training step [s]", title="(c) wall-clock time")
    for ax in (ax_m, ax_t):
        ax.legend(frameon=False, fontsize=8)
    return fig
