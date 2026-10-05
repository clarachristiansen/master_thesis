from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from neural_ode.data import TrajectoryData
from neural_ode.model import LatentODE


@dataclass
class TrainConfig:
    """Optimisation settings for :func:`train`.

    ``window_frames``: if set, each iteration trains on a random window of this
    many frames instead of the whole trajectory (see :func:`train`).
    """

    n_iters: int = 3000
    lr: float = 3e-3
    grad_clip: float = 1.0
    batch_size: int = 32
    eval_every: int = 50
    seed: int = 0
    window_frames: Optional[int] = None


@dataclass
class TrainHistory:
    """Losses recorded every ``eval_every`` iterations."""

    iters: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_iter: int = 0


def trajectory_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error over trajectories, frames and observed dimensions."""
    return torch.mean((pred - target) ** 2)


@torch.no_grad()
def reconstruction_loss(model: LatentODE, data: TrajectoryData) -> float:
    """:func:`trajectory_mse` of the model's reconstruction of every trajectory in ``data``."""
    times, obs = data.tensors()
    pred, _ = model(times, obs)
    return float(trajectory_mse(pred, obs))


def train(model: LatentODE, train_data: TrajectoryData, val_data: TrajectoryData, config: TrainConfig) -> TrainHistory:
    """Fit ``model`` by minimising the trajectory MSE of its reconstructions with Adam.

    Each iteration encodes a random mini-batch of trajectories, integrates
    their latent ODEs and decodes them. With ``config.window_frames`` set, the
    iteration uses a random window of that length (the same start for the
    whole batch) instead of the full time grid, re-timed to start at 0; the
    dynamics are autonomous, so only elapsed time matters. Windows that start
    one period apart contain the same data and must be encoded alike, which
    rewards latent paths that return to themselves (closed orbits) over
    one-off transients the decoder folds into oscillations. Validation always
    uses whole trajectories.
    Backpropagating through the whole unrolled solver on an oscillating
    signal occasionally produces very large gradients that throw the vector
    field far off; gradient-norm clipping and a cosine learning-rate decay
    guard against that. The parameters with the lowest validation loss are
    restored at the end.

    Returns:
        The loss history; ``model`` is modified in place.
    """
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    times, obs = train_data.tensors()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iters)
    history = TrainHistory()
    best_val, best_state = np.inf, copy.deepcopy(model.state_dict())

    for it in range(1, config.n_iters + 1):
        batch = rng.choice(len(train_data), size=min(config.batch_size, len(train_data)), replace=False)
        window = slice(None)
        if config.window_frames is not None:
            start = int(rng.integers(0, len(times) - config.window_frames + 1))
            window = slice(start, start + config.window_frames)
        target = obs[batch][:, window]
        pred, latents = model(times[window] - times[window][0], target)
        loss = trajectory_mse(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if it % config.eval_every == 0 or it == config.n_iters:
            val = reconstruction_loss(model, val_data)
            history.iters.append(it)
            history.train_loss.append(reconstruction_loss(model, train_data))
            history.val_loss.append(val)
            if val < best_val:
                best_val, best_state, history.best_iter = val, copy.deepcopy(model.state_dict()), it

    model.load_state_dict(best_state)
    return history
