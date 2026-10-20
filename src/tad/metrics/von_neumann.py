"""Threshold-free von Neumann entropy for a Pearson correlation matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import xlogy

from tad.metrics.correlation import CorrelationResult


@dataclass(frozen=True)
class VonNeumannEntropyResult:
    """Entropy values and spectrum of the Pearson density matrix."""

    entropy: float
    normalized_entropy: float
    eigenvalues: np.ndarray
    channels: list | None
    info: dict[str, Any]


def compute_von_neumann_entropy(
    pearson_correlation: np.ndarray | CorrelationResult,
) -> VonNeumannEntropyResult:
    """Compute threshold-free von Neumann entropy from a Pearson matrix.

    The paper's density matrix is ``rho = R / N``, where ``R`` is the full
    Pearson correlation matrix and ``N`` is its number of channels. Its entropy
    is ``S = -sum(lambda * log(lambda))``. The normalized value is ``S / log(N)``.

    Parameters
    ----------
    pearson_correlation
        A Pearson correlation matrix, or its TAD ``CorrelationResult``. The
        matrix must be symmetric, have unit diagonal, and be positive
        semidefinite. It is used without thresholding or removing correlations.

    Returns
    -------
    VonNeumannEntropyResult
        Entropy in nats, entropy normalized to [0, 1] by ``log(N)``, and the
        eigenvalues of ``rho`` after correcting only tiny negative roundoff.

    Notes
    -----
    This method is defined for Pearson correlation matrices. Do not pass mutual
    information or lagged cross-correlation matrices as substitutes.
    """
    channels = None
    source_info = {}
    if isinstance(pearson_correlation, CorrelationResult):
        matrix = np.asarray(pearson_correlation.correlation, dtype=float)
        channels = list(pearson_correlation.channels)
        source_info = dict(pearson_correlation.info)
    else:
        original = np.asarray(pearson_correlation)
        if np.iscomplexobj(original):
            raise ValueError("Pearson correlation matrix must be real")
        matrix = np.asarray(original, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Pearson correlation matrix must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Pearson correlation matrix must contain only finite values")

    n_channels = matrix.shape[0]
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-8):
        raise ValueError("Pearson correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Pearson correlation matrix must have a unit diagonal")

    # The Pearson matrix is already validated as symmetric; use rho = R / N.
    density_matrix = matrix / n_channels
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    tolerance = (
        100.0 * np.finfo(float).eps * n_channels
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if eigenvalues.min() < -tolerance:
        raise ValueError(
            "Pearson correlation matrix must be positive semidefinite; "
            "this matrix has a negative eigenvalue beyond numerical tolerance"
        )

    # Remove only tiny negative eigenvalues caused by floating-point roundoff.
    # Since density_matrix is R / n_channels and trace(R) = n_channels,
    # its eigenvalues already sum to one; do not renormalize them again.
    #eigenvalues = np.maximum(eigenvalues, 0.0)
    # xlogy(0, 0) is defined as 0, so this sums the full spectrum directly.
    entropy = float(-np.sum(xlogy(eigenvalues, eigenvalues)))
    max_entropy = float(np.log(n_channels)) if n_channels > 1 else 0.0
    normalized_entropy = (
        float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy else 0.0
    )

    return VonNeumannEntropyResult(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        eigenvalues=eigenvalues,
        channels=channels,
        info={
            **source_info,
            "n_channels": n_channels,
            "density_matrix": "Pearson correlation / n_channels",
            "log_base": "e",
            "maximum_entropy": max_entropy,
            "normalization": "entropy / log(n_channels)",
            "threshold_free": True,
        },
    )
