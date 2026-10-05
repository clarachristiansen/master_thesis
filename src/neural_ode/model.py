from __future__ import annotations

import torch
from torch import nn

from neural_ode.solvers import odeint_rk4


def mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    """Fully connected network with ``n_hidden`` tanh hidden layers (smooth, as a vector field should be)."""
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(dim, hidden_dim), nn.Tanh()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SkipDecoder(nn.Module):
    """Affine map plus an MLP correction: decoder(z) = W z + b + MLP(z).

    The MLP's output layer starts at zero, so the decoder is exactly affine at
    initialisation and training starts in the regime where the latent ODE,
    not the decoder, has to produce the oscillation. The MLP then adds
    nonlinear corrections; the decoder is as expressive as a plain MLP.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, obs_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, obs_dim)
        self.mlp = mlp(latent_dim, hidden_dim, obs_dim)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Observations (..., obs_dim) from latents (..., latent_dim)."""
        return self.linear(z) + self.mlp(z)


class Drift(nn.Module):
    """The learned vector field f_theta in dz/dt = f_theta(z, t).

    The network is autonomous: it takes ``t`` for interface compatibility with
    the solver (and a future latent SDE drift) but ignores it, matching the
    time-invariant ground-truth oscillator.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = mlp(latent_dim, hidden_dim, latent_dim)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """f_theta(z), shape (..., latent_dim)."""
        return self.net(z)


class GRUEncoder(nn.Module):
    """Maps an observed series to z0 by running a GRU backward in time.

    Reading the series from the last frame to the first means the final
    hidden state summarises the whole trajectory "as seen from" t0, the
    Latent ODE / Li et al. (2020) construction. The output is a point
    estimate of z0; a variational encoder would output a distribution.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode observations of shape (B, T, obs_dim) into z0 of shape (B, latent_dim)."""
        _, h = self.gru(observations.flip(1))
        return self.out(h[-1])


class LatentODE(nn.Module):
    """Encoder -> Neural ODE in latent space -> decoder.

    Generative path: z0 = encoder(x_{0:T}), z(t) = z0 + int_0^t f_theta(z) ds,
    x_hat(t) = decoder(z(t)). The three parts are separate modules so the
    deterministic drift can later be joined by a diffusion network and the
    encoder made variational, without changing the decoder or data pipeline.

    The default decoder is a :class:`SkipDecoder` (affine map plus an MLP
    correction that starts at zero). A plain MLP decoder, trained from a
    random initialisation, tended to find solutions in which the decoder
    folds a non-repeating latent path into oscillations; starting from an
    affine decoder avoids that basin without limiting expressivity.
    ``linear_decoder=True`` keeps only the affine map, restricting the latent
    space to affine images of the dynamics.
    """

    def __init__(
        self,
        obs_dim: int = 1,
        latent_dim: int = 2,
        hidden_dim: int = 64,
        encoder_hidden_dim: int = 32,
        steps_per_frame: int = 1,
        linear_decoder: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps_per_frame = steps_per_frame
        self.encoder = GRUEncoder(obs_dim, latent_dim, encoder_hidden_dim)
        self.drift = Drift(latent_dim, hidden_dim)
        self.decoder = (
            nn.Linear(latent_dim, obs_dim) if linear_decoder else SkipDecoder(latent_dim, hidden_dim, obs_dim)
        )

    def integrate(self, z0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Latent trajectories from z0 of shape (B, latent_dim); returns (B, T, latent_dim)."""
        return odeint_rk4(self.drift, z0, times, self.steps_per_frame).transpose(0, 1)

    def forward(self, times: torch.Tensor, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct observed trajectories through the latent ODE.

        Args:
            times: Frame times, shape (T,).
            observations: Observed series, shape (B, T, obs_dim).

        Returns:
            ``(predicted observations (B, T, obs_dim), latent trajectories (B, T, latent_dim))``.
        """
        latents = self.integrate(self.encoder(observations), times)
        return self.decoder(latents), latents
