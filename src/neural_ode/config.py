"""Experiment configuration: dataclasses with defaults, loaded from YAML plus command-line overrides.

A run is fully described by one :class:`ExperimentConfig`. Any field can be set
in a YAML file or overridden with a dotted key, for example::

    python -m neural_ode.run configs/base_harmonic.yaml data.n_train=512 seed=1

Fields left out of the YAML keep their defaults below.
"""

from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neural_ode.train import TrainConfig


@dataclass
class DataConfig:
    system: str = "harmonic_oscillator"  # key in neural_ode.run.SYSTEMS
    system_kwargs: dict[str, Any] = field(default_factory=dict)  # arguments of the system, e.g. its priors
    t_end: float = 30.0  # s, length of the training window
    dt: float = 0.5  # s per frame
    n_train: int = 128
    n_val: int = 32
    n_test: int = 32
    seed: int = 0  # keep fixed: the top-level seed varies model init and batch order only

    def times(self) -> np.ndarray:
        return np.arange(0.0, self.t_end + 1e-9, self.dt)


@dataclass
class ModelConfig:
    latent_dim: int = 2
    decoder: str = "mlp"  # "mlp"
    drift_hidden_dim: int = 64  # hidden width of the drift MLP
    steps_per_frame: int = 1  # RK4 sub-steps per frame interval
    gradient: str = "backprop"  # "backprop" through the solver, or "adjoint" (neural_ode.solvers)


@dataclass
class EvalConfig:
    # Forecast trajectories run to this time; the encoder sees only the first data.t_end seconds.
    # A value <= data.t_end switches forecasting off.
    forecast_t_end: float = 60.0
    n_forecast: int = 32


@dataclass
class PlotConfig:
    """Figure settings only. They change no metric, so they are left out of the W&B config."""

    color_param: str | None = None  # true parameter that colours the plots
    # Parameter grid pushed through the model for the latent-flow plot; None skips the plot.
    # Exactly two entries. A list [start, stop, num] means linspace(start, stop, num);
    # a single number n means an angle: n points evenly round the circle.
    grid: dict[str, Any] | None = None
    snapshot_times: list[float] = field(default_factory=lambda: [0.0, 2.5, 5.0, 10.0])  # s


@dataclass
class ExperimentConfig:
    name: str = "neural_ode"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    out_dir: str = "models"  # model, config and metrics: out_dir/name/run_id
    fig_dir: str = "reports/figures"  # figures: fig_dir/name/run_id
    wandb_project: str | None = "Master's Thesis"  # None: local files only
    tags: list[str] = field(default_factory=list)

    def tracked(self) -> dict[str, Any]:
        """The settings that define the experiment: what W&B stores as the run's config."""
        d = to_dict(self)
        for key in ("plots", "out_dir", "fig_dir", "wandb_project", "tags"):
            d.pop(key)
        return d

    def group(self) -> str:
        """Identifier shared by runs that differ only in seed, used to group seeds in W&B."""
        d = self.tracked()
        d.pop("seed")
        d["train"].pop("seed", None)
        digest = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        return f"{self.name}-{digest}"


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain nested dict of ``cfg`` (tuples become lists, so it is safe for YAML and JSON)."""
    return json.loads(json.dumps(asdict(cfg)))


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))


def load_config(path: str | Path | None = None, overrides: typing.Iterable[str] = ()) -> ExperimentConfig:
    """Defaults, updated by the YAML file at ``path`` (if given), then by ``key=value`` overrides."""
    values = (yaml.safe_load(Path(path).read_text()) or {}) if path else {}
    return _build(ExperimentConfig, apply_overrides(values, overrides))


def apply_overrides(values: dict[str, Any], overrides: typing.Iterable[str]) -> dict[str, Any]:
    """Set dotted ``key=value`` items in the nested dict ``values`` (leading dashes are ignored)."""
    for item in overrides:
        key, sep, raw = item.lstrip("-").partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key=value")
        *parents, leaf = key.split(".")
        node = values
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = _parse(raw)
    return values


def _parse(raw: str) -> Any:
    """YAML scalar, but with '1e-3' as a float (PyYAML reads it as a string)."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Instantiate the dataclass ``cls`` from a nested dict; unknown keys are an error (catches typos)."""
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        kwargs[name] = _build(hint, value) if is_dataclass(hint) and isinstance(value, dict) else value
    return cls(**kwargs)
