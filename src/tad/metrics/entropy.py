"""Shannon entropy of the degree distribution of a thresholded network."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShannonEntropyResult:
    """Shannon entropy and the degree distribution used to calculate it."""

    entropy: float
    degrees: np.ndarray
    degree_values: np.ndarray
    degree_probabilities: np.ndarray
    adjacency: np.ndarray
    threshold: float
    log_base: float


def compute_shannon_entropy(
    matrix: np.ndarray,
    threshold: float,
    log_base: float = 2.0,
) -> ShannonEntropyResult:
    """Threshold a symmetric connectivity matrix and calculate degree entropy.

    An undirected edge is present when ``abs(matrix[i, j]) > threshold``.
    Self-connections are excluded. The degree ``k_i`` is calculated for each
    node, then the degree distribution is

    ``P(k) = number of nodes with degree k / total number of nodes``.

    Shannon entropy is calculated from that distribution as
    ``-sum(P(k) * log(P(k)))``. The matrix is not modified in place, and this
    function does not select a threshold or sweep over multiple thresholds.

    Parameters
    ----------
    matrix
        A non-empty, real-valued, symmetric square connectivity matrix.
    threshold
        Non-negative absolute-value cutoff. Edges at exactly the cutoff are
        excluded, following the common ``abs(weight) > threshold`` rule.
    log_base
        Logarithm base. Defaults to 2, so entropy is returned in bits.

    Returns
    -------
    ShannonEntropyResult
        Entropy plus the node degrees, observed degree values, their
        probabilities, and the binary adjacency matrix used for the metric.
    """
    original = np.asarray(matrix)
    if original.ndim != 2 or original.shape[0] != original.shape[1] or original.size == 0:
        raise ValueError("matrix must be a non-empty square 2D array")
    if np.iscomplexobj(original):
        raise ValueError("matrix must contain real values")

    values = np.asarray(original, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not np.isfinite(log_base) or log_base <= 1:
        raise ValueError("log_base must be finite and greater than 1")
    if not np.allclose(values, values.T, rtol=0.0, atol=1e-12):
        raise ValueError("matrix must be symmetric for an undirected degree distribution")

    adjacency = (np.abs(values) > threshold).astype(np.uint8)
    np.fill_diagonal(adjacency, 0)
    degrees = adjacency.sum(axis=1, dtype=np.int64)
    degree_values, counts = np.unique(degrees, return_counts=True)
    degree_probabilities = counts.astype(np.float64) / degrees.size
    entropy = float(
        -np.sum(degree_probabilities * np.log(degree_probabilities))
        / np.log(log_base)
    )

    return ShannonEntropyResult(
        entropy=entropy,
        degrees=degrees,
        degree_values=degree_values,
        degree_probabilities=degree_probabilities,
        adjacency=adjacency,
        threshold=float(threshold),
        log_base=float(log_base),
    )
