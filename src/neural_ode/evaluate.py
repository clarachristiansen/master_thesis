from __future__ import annotations

from dataclasses import dataclass

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
class FitMetrics:
    """How well predicted observations match the ground truth.

    ``nmse`` is the MSE divided by the variance of the target, so 1 means
    "no better than predicting the mean" and 0 is a perfect fit.
    """

    mse: float
    rmse: float
    nmse: float
    per_trajectory_mse: np.ndarray  # (N,)
    per_frame_mse: np.ndarray  # (T,)


@torch.no_grad()
def predict(model: LatentODE, data: TrajectoryData) -> Prediction:
    """Encode, integrate and decode every trajectory in ``data``."""
    times, obs = data.tensors()
    pred, latents = model(times, obs)
    return Prediction(pred.numpy(), latents.numpy())


@torch.no_grad()
def encode(model: LatentODE, data: TrajectoryData) -> np.ndarray:
    """Encoded initial states z0 of every trajectory in ``data``, shape (N, latent_dim)."""
    return model.encoder(torch.as_tensor(data.observations, dtype=torch.float32)).numpy()


def fit_metrics(pred: np.ndarray, target: np.ndarray) -> FitMetrics:
    """Error metrics between predicted and target observations, both of shape (N, T, obs_dim)."""
    err = (pred - target) ** 2
    mse = float(err.mean())
    return FitMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        nmse=mse / float(target.var()),
        per_trajectory_mse=err.mean(axis=(1, 2)),
        per_frame_mse=err.mean(axis=(0, 2)),
    )


@dataclass
class OrbitFrame:
    """Orthonormal latent coordinates adapted to a family of orbits.

    Columns 0 and 1 of ``basis`` span the rotation plane, the plane in which
    trajectories circulate. Column 2 (only if latent_dim >= 3) is the
    direction orthogonal to that plane along which the orbit centres spread
    most: where a cylinder-like latent space stores omega. Signs of the
    columns are arbitrary.
    """

    basis: np.ndarray  # (latent_dim, min(latent_dim, 3))
    origin: np.ndarray  # (latent_dim,) mean of all latents

    def project(self, latents: np.ndarray) -> np.ndarray:
        """Coordinates (u, v[, w]) of latents (..., latent_dim) in this frame."""
        return (latents - self.origin) @ self.basis


def orbit_frame(latents: np.ndarray) -> OrbitFrame:
    """:class:`OrbitFrame` of latent trajectories (N, T, latent_dim).

    The rotation plane is the top-2 principal plane of the latents after
    centring each trajectory on its own centroid, which removes the offsets
    between orbits and keeps the circulation. The third axis is the top
    principal direction of the orbit centroids within the orthogonal complement.
    """
    dim = latents.shape[-1]
    centred = latents - latents.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(centred.reshape(-1, dim), full_matrices=True)
    plane = vt[:2].T
    if dim < 3:
        return OrbitFrame(plane, latents.reshape(-1, dim).mean(axis=0))
    complement = vt[2:].T
    centroids = latents.mean(axis=1)
    spread = (centroids - centroids.mean(axis=0)) @ complement
    _, _, vt_c = np.linalg.svd(spread, full_matrices=True)
    axis = complement @ vt_c[0]
    return OrbitFrame(np.column_stack([plane, axis]), latents.reshape(-1, dim).mean(axis=0))


@dataclass
class OrbitCoordinates:
    """Per-trajectory normalised coordinates in the rotation plane.

    Each trajectory's projection on the rotation plane is fitted with an
    ellipse, and mapped by an orientation-preserving (symmetric) linear map
    onto the unit circle centred at 0. Any orbit that is an affine image of a
    circle therefore becomes the unit circle, however much of it was sampled,
    and the sense of rotation stays comparable across trajectories.
    """

    plane: np.ndarray  # (latent_dim, 2)
    centroids: np.ndarray  # (N, latent_dim) orbit centres: ellipse centre in-plane, mean out-of-plane
    transforms: np.ndarray  # (N, 2, 2)

    def __call__(self, latents: np.ndarray) -> np.ndarray:
        """Normalised coordinates of latents (N, T, latent_dim) of the same N trajectories; returns (N, T, 2)."""
        return np.einsum("ntd,nde->nte", (latents - self.centroids[:, None]) @ self.plane, self.transforms)

    def radius(self, latents: np.ndarray) -> np.ndarray:
        """Normalised distance from each trajectory's orbit centre, shape (N, T); ~1 on its own orbit."""
        return np.linalg.norm(self(latents), axis=-1)


def _fit_ellipse(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre (2,) and symmetric map (2, 2) taking the ellipse through points (T, 2) to the unit circle.

    Least-squares conic fit a x^2 + b xy + c y^2 + d x + e y = 1 (points are
    centred first for conditioning). If the fitted conic is not an ellipse,
    falls back to the points' mean and covariance.
    """
    mean = pts.mean(axis=0)
    x, y = (pts - mean).T
    p, *_ = np.linalg.lstsq(np.column_stack([x**2, x * y, y**2, x, y]), np.ones(len(pts)), rcond=None)
    q = np.array([[p[0], p[1] / 2], [p[1] / 2, p[2]]])
    evals, evecs = np.linalg.eigh(q)
    if np.all(evals > 0):
        centre = -0.5 * np.linalg.solve(q, p[3:])
        k = 1.0 + centre @ q @ centre
        if k > 0:
            return mean + centre, evecs @ np.diag(np.sqrt(evals / k)) @ evecs.T
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    inv_sqrt = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T
    return mean, inv_sqrt / np.mean(np.linalg.norm((pts - mean) @ inv_sqrt, axis=-1))


def orbit_coordinates(latents: np.ndarray, frame: OrbitFrame) -> OrbitCoordinates:
    """:class:`OrbitCoordinates` fitted to latent trajectories (N, T, latent_dim) in ``frame``."""
    plane = frame.basis[:, :2]
    means = latents.mean(axis=1)
    centroids, transforms = [], []
    for mean, pts in zip(means, (latents - means[:, None]) @ plane):
        centre, transform = _fit_ellipse(pts)
        centroids.append(mean + plane @ centre)
        transforms.append(transform)
    return OrbitCoordinates(plane, np.stack(centroids), np.stack(transforms))


def winding_frequencies(coords: np.ndarray, times: np.ndarray) -> np.ndarray:
    """|d angle / dt| of each trajectory in :class:`OrbitCoordinates` (N, T, 2), by least squares; shape (N,)."""
    angle = np.unwrap(np.arctan2(coords[..., 1], coords[..., 0]), axis=1)
    return np.abs(np.polyfit(times, angle.T, 1)[0])


def initial_phases(coords: np.ndarray) -> np.ndarray:
    """Angle in [0, 2 pi) of each trajectory's first latent point in :class:`OrbitCoordinates`; shape (N,)."""
    return np.mod(np.arctan2(coords[:, 0, 1], coords[:, 0, 0]), 2 * np.pi)


@dataclass
class PhaseFit:
    """Best circular-linear fit latent_phase = orientation * true_phase + offset (mod 2 pi).

    Attributes:
        orientation: +1 or -1, the sense of rotation relative to the true phase.
        offset: Rotation between latent and true phase, in rad.
        rms_error: Root-mean-square wrapped residual in rad; 0 means the latent
            phase is an exact rotation/reflection of the true phase, random
            phases give about 1.8.
    """

    orientation: int
    offset: float
    rms_error: float


def fit_phase(latent: np.ndarray, true: np.ndarray) -> PhaseFit:
    """How well latent phases (N,) are explained as a rotation or reflection of true phases (N,)."""
    fits = []
    for orientation in (1, -1):
        diff = latent - orientation * true
        offset = float(np.angle(np.mean(np.exp(1j * diff))))
        residual = np.angle(np.exp(1j * (diff - offset)))
        fits.append(PhaseFit(orientation, offset, float(np.sqrt(np.mean(residual**2)))))
    return min(fits, key=lambda f: f.rms_error)


def fit_phase_per_omega(latent: np.ndarray, true: np.ndarray, omegas: np.ndarray) -> dict[float, PhaseFit]:
    """:func:`fit_phase` separately for every distinct omega (e.g. of a :func:`~neural_ode.data.parameter_grid`).

    A separate offset per omega is needed because rotating each orbit by an
    omega-dependent angle leaves every prediction unchanged, so the data
    cannot fix a common offset.
    """
    return {float(w): fit_phase(latent[omegas == w], true[omegas == w]) for w in np.unique(omegas)}


@dataclass
class LatentDiagnostics:
    """Frequency recovery of the learned latent orbits, per test trajectory.

    Attributes:
        frame: The :class:`OrbitFrame` the orbits are expressed in.
        coords: Normalised per-trajectory coordinates (:class:`OrbitCoordinates`).
        omega_true: True omega of each trajectory, shape (N,).
        omega_hat: Winding frequency of each latent trajectory, shape (N,).
    """

    frame: OrbitFrame
    coords: OrbitCoordinates
    omega_true: np.ndarray
    omega_hat: np.ndarray

    def summary(self) -> dict[str, float]:
        """Median relative error of omega_hat, and R^2 of omega_hat against the true omega (identity line)."""
        rel_err = np.abs(self.omega_hat - self.omega_true) / self.omega_true
        ss_res = np.sum((self.omega_hat - self.omega_true) ** 2)
        ss_tot = np.sum((self.omega_true - self.omega_true.mean()) ** 2)
        return {"omega_rel_error": float(np.median(rel_err)), "omega_hat_r2": float(1.0 - ss_res / ss_tot)}


def latent_diagnostics(prediction: Prediction, data: TrajectoryData) -> LatentDiagnostics:
    """:class:`LatentDiagnostics` for one model's predictions on an oscillator dataset (``data.params`` has "omega")."""
    latents = prediction.latents
    frame = orbit_frame(latents)
    coords = orbit_coordinates(latents, frame)
    return LatentDiagnostics(
        frame=frame,
        coords=coords,
        omega_true=data.params["omega"],
        omega_hat=winding_frequencies(coords(latents), data.times),
    )
