from __future__ import annotations

from typing import Callable

import torch

VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def rk4_step(func: VectorField, t: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One classical Runge-Kutta step, the torch twin of :class:`cell_simulator.integrators.RK4`."""
    k1 = func(t, z)
    k2 = func(t + h / 2, z + h / 2 * k1)
    k3 = func(t + h / 2, z + h / 2 * k2)
    k4 = func(t + h, z + h * k3)
    return z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_rk4(func: VectorField, z0: torch.Tensor, times: torch.Tensor, steps_per_frame: int = 1) -> torch.Tensor:
    """Integrate dz/dt = func(t, z) with fixed-step RK4, differentiably.

    Gradients flow by ordinary backpropagation through the solver steps (no
    adjoint method), which is cheap at this problem size. As in
    :meth:`cell_simulator.simulator.Experiment.run`, each frame interval is
    split into ``steps_per_frame`` sub-steps and only frame states are returned.

    Args:
        func: Vector field ``func(t, z)``; ``t`` is a scalar tensor, ``z`` has shape (..., D).
        z0: Initial state at ``times[0]``, shape (..., D).
        times: Increasing frame times, shape (T,).
        steps_per_frame: Solver sub-steps per frame interval.

    Returns:
        States at every frame time, shape (T, ..., D) (time-first, like ``Integrator.integrate``).
    """
    if steps_per_frame < 1:
        raise ValueError("steps_per_frame must be >= 1")
    states = [z0]
    z = z0
    for k in range(len(times) - 1):
        h = (times[k + 1] - times[k]) / steps_per_frame
        for s in range(steps_per_frame):
            z = rk4_step(func, times[k] + s * h, z, h)
        states.append(z)
    return torch.stack(states)
