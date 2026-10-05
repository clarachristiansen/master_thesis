from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from cell_simulator.dynamics import HarmonicOscillatorContraction
from cell_simulator.integrators import RK4

Params = dict[str, np.ndarray]


class System(ABC):
    """A dynamical system that generates synthetic trajectories for training and evaluation.

    A child defines its equations, dx = drift(t, x) dt + diffusion(t, x) dW, how
    parameters and initial states are drawn, and what is observed. The shared
    :meth:`simulate` integrates all trajectories at once with Euler-Maruyama
    (plain Euler when :meth:`diffusion` returns None, i.e. for an ODE).
    Noise is diagonal: each state dimension gets its own independent Wiener process.

    Every parameter is per trajectory: ``params`` maps a name to an array of
    shape (N,), so trajectories may differ in their parameters.
    """

    state_dim: int
    obs_dim: int

    @abstractmethod
    def sample_params(self, n: int, rng: np.random.Generator) -> Params:
        """Parameters of ``n`` trajectories, each value of shape (n,)."""

    @abstractmethod
    def initial_state(self, params: Params, rng: np.random.Generator) -> np.ndarray:
        """Initial states, shape (N, state_dim)."""

    @abstractmethod
    def drift(self, t: float, x: np.ndarray, params: Params) -> np.ndarray:
        """Drift at states x of shape (N, state_dim); returns (N, state_dim)."""

    def diffusion(self, t: float, x: np.ndarray, params: Params) -> Optional[np.ndarray]:
        """Diagonal diffusion at states x (N, state_dim), same shape; None for a deterministic system."""
        return None

    @abstractmethod
    def observe(self, states: np.ndarray, params: Params, rng: np.random.Generator) -> np.ndarray:
        """Observations of states (N, T, state_dim); returns (N, T, obs_dim)."""

    def simulate(
        self, params: Params, times: np.ndarray, rng: np.random.Generator, steps_per_frame: int = 10
    ) -> np.ndarray:
        """Integrate all trajectories over ``times`` with Euler-Maruyama.

        Each frame interval is split into ``steps_per_frame`` sub-steps and only
        frame states are returned, as in :meth:`cell_simulator.simulator.Experiment.run`.

        Returns:
            States at the frame times, shape (N, T, state_dim).
        """
        x = self.initial_state(params, rng)
        states = [x]
        for k in range(len(times) - 1):
            h = (times[k + 1] - times[k]) / steps_per_frame
            for s in range(steps_per_frame):
                t = times[k] + s * h
                g = self.diffusion(t, x, params)
                x = x + h * self.drift(t, x, params)
                if g is not None:
                    x = x + g * np.sqrt(h) * rng.standard_normal(x.shape)
            states.append(x)
        return np.stack(states, axis=1)


class HarmonicOscillator(System):
    """The harmonic-oscillator contraction clock of :mod:`cell_simulator`, observed through a(t).

    State (c, z), parameters omega (rad/s) and phase (rad, in [0, 2 pi)),
    observation a = (1 + c) / 2. Deterministic, so :meth:`simulate` uses
    :class:`cell_simulator.integrators.RK4` on
    :class:`cell_simulator.dynamics.HarmonicOscillatorContraction` instead of
    Euler-Maruyama. Defaults are the priors of ``notebooks/cell_simulator_MC.ipynb``:
    omega ~ |N(0.6, 0.15^2)|, phase ~ N(0, 1.5^2) wrapped to [0, 2 pi).
    """

    state_dim = 2
    obs_dim = 1

    def __init__(self, omega_mean: float = 0.6, omega_std: float = 0.15, phase_std: float = 1.5):
        self.omega_mean, self.omega_std, self.phase_std = omega_mean, omega_std, phase_std

    def sample_params(self, n: int, rng: np.random.Generator) -> Params:
        omega = np.abs(rng.normal(self.omega_mean, self.omega_std, n))
        phase = np.mod(rng.normal(0.0, self.phase_std, n), 2 * np.pi)
        return {"omega": omega, "phase": phase}

    def initial_state(self, params: Params, rng: np.random.Generator) -> np.ndarray:
        return np.stack([np.sin(params["phase"]), np.cos(params["phase"])], axis=-1)

    def drift(self, t: float, x: np.ndarray, params: Params) -> np.ndarray:
        omega = params["omega"]
        return np.stack([omega * x[:, 1], -omega * x[:, 0]], axis=-1)

    def observe(self, states: np.ndarray, params: Params, rng: np.random.Generator) -> np.ndarray:
        return 0.5 * (1.0 + states[..., :1])

    def simulate(
        self, params: Params, times: np.ndarray, rng: np.random.Generator, steps_per_frame: int = 10
    ) -> np.ndarray:
        """RK4 per trajectory with the simulator's own classes (see class docstring)."""
        n_steps = (len(times) - 1) * steps_per_frame
        states = []
        for omega, phase in zip(params["omega"], params["phase"]):
            dynamics = HarmonicOscillatorContraction(omega, phase)
            _, fine = RK4().integrate(dynamics, times[0], times[-1], n_steps, dynamics.initial_state())
            states.append(fine[::steps_per_frame])
        return np.stack(states)

    @staticmethod
    def periods(params: Params) -> np.ndarray:
        """True oscillation period 2 pi / omega of each trajectory, in s."""
        return 2 * np.pi / params["omega"]


# class OrnsteinUhlenbeck(System)
# class StochasticLorenz(System)
