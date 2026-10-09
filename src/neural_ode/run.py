"""Train, evaluate and log one experiment.

    python -m neural_ode.run configs/base.yaml data.n_train=512 seed=1

A run saves
- ``out_dir/name/run_id/``: ``config.yaml``, ``model.pt`` and ``metrics.json``;
- ``fig_dir/name/run_id/``: one PNG per figure.

``run_id`` is the W&B run id (also used as the W&B run name), or a timestamp
without W&B. W&B gets the same metrics and figures, the training curves, and
the model checkpoint as an artifact.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_ode.config import ExperimentConfig, load_config, save_config, to_dict
from neural_ode.data import TrajectoryData, make_splits, parameter_grid
from neural_ode.evaluate import GridAxis, evaluate, make_figures
from neural_ode.model import LatentODE
from neural_ode.system import HarmonicOscillator
from neural_ode.train import TrainHistory, train

SYSTEMS = {
    "harmonic_oscillator": HarmonicOscillator,
    # "damped_oscillator": DampedOscillator,
}


def build_data(cfg: ExperimentConfig) -> tuple[dict[str, TrajectoryData], TrajectoryData | None, Any]:
    """Train/val/test splits, longer forecast trajectories with fresh parameters (or None), and the system."""
    system = SYSTEMS[cfg.data.system]()
    d = cfg.data
    splits = make_splits(system, d.times(), d.n_train, d.n_val, d.n_test, seed=d.seed)
    forecast = None
    if cfg.eval.forecast_t_end > d.t_end:
        long_times = np.arange(0.0, cfg.eval.forecast_t_end + 1e-9, d.dt)
        # A different data seed, so the forecast parameters are new draws from the same priors.
        forecast = make_splits(system, long_times, 1, 1, cfg.eval.n_forecast, seed=d.seed + 1000)["test"]
    return splits, forecast, system


def build_grid(cfg: ExperimentConfig, system: Any) -> tuple[TrajectoryData | None, list[GridAxis] | None]:
    """Trajectories on the parameter grid of ``plots.grid`` (see PlotConfig), or (None, None)."""
    spec = cfg.plots.grid
    if not spec:
        return None, None
    if len(spec) != 2:
        raise ValueError(f"plots.grid needs exactly two parameters, got {list(spec)}")
    axes = []
    for name, value in spec.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            axes.append(GridAxis(name, np.linspace(0.0, 2 * np.pi, int(value), endpoint=False), periodic=True))
        else:
            start, stop, num = value
            axes.append(GridAxis(name, np.linspace(start, stop, int(num)), periodic=False))
    grid = parameter_grid(system, cfg.data.times(), {axis.name: axis.values for axis in axes})
    return grid, axes


def build_model(cfg: ExperimentConfig) -> LatentODE:
    torch.manual_seed(cfg.seed)
    kwargs = {
        "latent_dim": cfg.model.latent_dim,
        "drift_hidden_dim": cfg.model.drift_hidden_dim,
        "steps_per_frame": cfg.model.steps_per_frame,
        "gradient": cfg.model.gradient,
    }
    return LatentODE(**kwargs)


def run(cfg: ExperimentConfig) -> dict[str, float]:
    """Train one model on ``cfg``, evaluate it, save everything and log it to W&B. Returns the metrics."""
    wandb_run = None
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if cfg.wandb_project:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, group=cfg.group(), tags=cfg.tags, config=cfg.tracked())
        run_id = wandb_run.id
        wandb_run.name = run_id  # the same name as the output folders
    out = Path(cfg.out_dir) / cfg.name / run_id
    fig_dir = Path(cfg.fig_dir) / cfg.name / run_id
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    splits, forecast, system = build_data(cfg)
    grid, grid_axes = build_grid(cfg, system)
    context_frames = len(cfg.data.times())
    model = build_model(cfg)
    history = train(model, splits["train"], splits["val"], replace(cfg.train, seed=cfg.seed))
    torch.save({"model": model.state_dict(), "history": asdict(history), "config": to_dict(cfg)}, out / "model.pt")

    metrics = evaluate(model, splits, forecast, context_frames=context_frames)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["model/n_params"] = sum(
        p.numel() for p in model.parameters()
    )  # added for sweeps, not in the original metrics.json

    figures = make_figures(
        model,
        splits,
        forecast,
        grid,
        grid_axes,
        context_frames=context_frames,
        color_param=cfg.plots.color_param,
        snapshot_times=cfg.plots.snapshot_times,
    )
    figures["history"] = plot_history(history)
    for name, fig in figures.items():
        fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")

    if wandb_run is not None:
        import wandb

        wandb.define_metric("history/iter")
        wandb.define_metric("history/*", step_metric="history/iter")
        for row in _history_rows(history):
            wandb.log(row)
        wandb.log({f"figures/{name}": wandb.Image(fig) for name, fig in figures.items()})
        wandb_run.summary.update(metrics)

        artifact = wandb.Artifact(f"{cfg.name}-{run_id}", type="model", metadata=metrics)
        artifact.add_file(str(out / "model.pt"))
        artifact.add_file(str(out / "config.yaml"))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()
    for fig in figures.values():
        plt.close(fig)
    return metrics


def plot_history(history: TrainHistory) -> plt.Figure:
    """Training and validation loss against iteration."""
    fig, ax = plt.subplots(figsize=(5, 3.4), layout="constrained")
    for name, (x, y) in _history_curves(history).items():
        ax.semilogy(x, y, label=name)
    ax.set(xlabel="iteration", ylabel="MSE", title="training history")
    ax.legend(frameon=False, fontsize=8)
    return fig


def _history_curves(history: TrainHistory) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loss curves of the history as (iteration, value) pairs.

    ``history.iters`` holds the iterations at which losses were recorded: it is the x-axis,
    not a curve. A curve of a different length is taken to be recorded every iteration.
    """
    h = asdict(history)
    iters = h.get("iters")
    curves = {}
    for name, values in h.items():
        if name == "iters" or not _is_curve(values):
            continue
        x = np.asarray(iters) if iters is not None and len(iters) == len(values) else np.arange(len(values))
        curves[name] = (x, np.asarray(values, dtype=float))
    return curves


def _history_rows(history: TrainHistory) -> list[dict[str, float]]:
    """Training curves as one dict per iteration, for wandb.log with ``history/iter`` as the x-axis."""
    rows: dict[int, dict[str, float]] = {}
    for name, (x, y) in _history_curves(history).items():
        for xi, yi in zip(x, y):
            rows.setdefault(int(xi), {})[f"history/{name}"] = float(yi)
    return [{"history/iter": i, **rows[i]} for i in sorted(rows)]


def _is_curve(values: object) -> bool:
    return isinstance(values, (list, tuple)) and len(values) > 0 and isinstance(values[0], (int, float))


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no windows when run from the command line
    try:
        from cell_simulator.visualize import set_style

        set_style()
    except ImportError:
        pass

    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args and args[0].endswith((".yaml", ".yml")) else None
    cfg = load_config(path, args[1:] if path else args)
    metrics = run(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
