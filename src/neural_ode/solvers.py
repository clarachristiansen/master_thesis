"""Fixed-step RK4 for the latent ODE, with two ways of computing gradients.

- ``gradient="backprop"`` (:func:`odeint_rk4`): autograd records every RK4 stage
  of the forward solve and backpropagates through them. This is the exact
  gradient of the discretised computation ("discretise-then-optimise"); memory
  grows with the number of solver steps.
- ``gradient="adjoint"`` (:func:`odeint_rk4_adjoint`): the forward solve stores
  only the frame states, and the gradient is obtained by solving the adjoint
  ODE backwards in time with the same RK4 steps ("optimise-then-discretise",
  Chen et al. 2018). It approximates the gradient of the continuous problem,
  so it agrees with backprop only up to the solver error and converges to it
  as the step size shrinks.

Both use the same forward solver, so predictions are identical; only the
gradients differ. :func:`odeint` switches between them.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.autograd.function import once_differentiable

VectorField = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GRADIENT_METHODS = ("backprop", "adjoint")


def rk4_step(func: VectorField, t: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """One classical Runge-Kutta step, the torch twin of :class:`cell_simulator.integrators.RK4`."""
    k1 = func(t, z)
    k2 = func(t + h / 2, z + h / 2 * k1)
    k3 = func(t + h / 2, z + h / 2 * k2)
    k4 = func(t + h, z + h * k3)
    return z + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def odeint_rk4(func: VectorField, z0: torch.Tensor, times: torch.Tensor, steps_per_frame: int = 1) -> torch.Tensor:
    """Integrate dz/dt = func(t, z) forward in time with fixed-step RK4.

    Written in torch operations only, so when a loss computed from the result
    is backpropagated, gradients reach the parameters of ``func`` through every
    solver step (plain backpropagation, not the adjoint method). As in
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


def odeint_rk4_adjoint(
    func: nn.Module, z0: torch.Tensor, times: torch.Tensor, steps_per_frame: int = 1
) -> torch.Tensor:
    """:func:`odeint_rk4` with gradients from the continuous adjoint method instead of backpropagation.

    Forward, the same RK4 solve runs without recording a graph, and only the
    frame states are kept. Backward, the augmented system

        dz/dt = f(z, t),   da/dt = -a df/dz,   dg/dt = -a df/dtheta

    is integrated from each frame to the previous one with the same RK4 steps
    (negative step size), where a(t) = dL/dz(t) is the adjoint and
    g accumulates dL/dtheta. Two details:

    - At every frame t_i the adjoint jumps by the loss gradient there,
      a <- a + dL/dz(t_i), which is how a loss on the whole series enters.
    - At every frame z is reset to the stored forward state, so z is only
      reconstructed backwards within one frame interval. Run backwards, an
      attracting orbit repels, and reconstructing z over the whole series
      would let errors grow.

    a(t_0) is returned as the gradient for ``z0``, so an encoder that produced
    ``z0`` is trained as usual.

    Args:
        func: Vector field ``func(t, z)`` as an ``nn.Module``; gradients are computed for
            its ``parameters()`` that require grad, and for nothing else it closes over.
        z0: Initial state at ``times[0]``, shape (..., D).
        times: Increasing frame times, shape (T,).
        steps_per_frame: Solver sub-steps per frame interval, forward and backward.

    Returns:
        States at every frame time, shape (T, ..., D), identical to :func:`odeint_rk4`.
    """
    if steps_per_frame < 1:
        raise ValueError("steps_per_frame must be >= 1")
    if not isinstance(func, nn.Module):
        raise TypeError("the adjoint method needs func to be an nn.Module, so its parameters can be found")
    params = tuple(p for p in func.parameters() if p.requires_grad)
    return _RK4Adjoint.apply(func, steps_per_frame, z0, times, *params)


def odeint(
    func: nn.Module, z0: torch.Tensor, times: torch.Tensor, steps_per_frame: int = 1, gradient: str = "backprop"
) -> torch.Tensor:
    """Fixed-step RK4 solve; ``gradient`` ("backprop" or "adjoint") picks how gradients are computed."""
    if gradient == "backprop":
        return odeint_rk4(func, z0, times, steps_per_frame)
    if gradient == "adjoint":
        return odeint_rk4_adjoint(func, z0, times, steps_per_frame)
    raise ValueError(f"gradient must be one of {GRADIENT_METHODS}, got {gradient!r}")


class _RK4Adjoint(torch.autograd.Function):
    """Autograd wrapper behind :func:`odeint_rk4_adjoint`."""

    @staticmethod
    def forward(ctx, func, steps_per_frame, z0, times, *params):
        with torch.no_grad():
            zs = odeint_rk4(func, z0, times, steps_per_frame)
        ctx.func, ctx.steps_per_frame, ctx.params = func, steps_per_frame, params
        ctx.save_for_backward(zs, times)
        return zs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_zs):
        zs, times = ctx.saved_tensors
        func, steps_per_frame, params = ctx.func, ctx.steps_per_frame, ctx.params

        shapes = [zs[0].shape, zs[0].shape, *(p.shape for p in params)]
        sizes = [shape.numel() for shape in shapes]

        def unflatten(state: torch.Tensor) -> list[torch.Tensor]:
            return [x.view(shape) for x, shape in zip(state.split(sizes), shapes)]

        def augmented(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            z, a, *_ = unflatten(state)
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                f = func(t, z)
                vjp = torch.autograd.grad(f, (z, *params), grad_outputs=a, allow_unused=True)
            dg = [-v if v is not None else torch.zeros_like(p) for v, p in zip(vjp[1:], params)]
            return torch.cat([x.reshape(-1) for x in (f.detach(), -vjp[0], *dg)])

        a = grad_zs[-1]
        g = [torch.zeros_like(p) for p in params]
        for i in range(len(times) - 1, 0, -1):
            h = (times[i] - times[i - 1]) / steps_per_frame
            state = torch.cat([x.reshape(-1) for x in (zs[i], a, *g)])
            for s in range(steps_per_frame):
                state = rk4_step(augmented, times[i] - s * h, state, -h)
            _, a, *g = unflatten(state)
            a = a + grad_zs[i - 1]
        return (None, None, a, None, *g)
