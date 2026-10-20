"""Descriptive graph metrics from a connectivity matrix.

Weights represent connection strengths, not distances or statistical significance.
Directed input uses the convention matrix[i, j] = i -> j. Self-loops are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class NetworkMetricsResult:
    """Graph, node-aligned measures, global measures, and explicit conventions."""

    channels: list
    adjacency: np.ndarray
    node_metrics: dict[str, np.ndarray]
    global_metrics: dict[str, float | int]
    info: dict[str, Any]


def _clustering(weights, directed):
    """Vectorized Onnela / Fagiolo clustering (same normalization as NetworkX)."""
    a = (weights > 0).astype(float)
    scale = float(weights.max(initial=0))
    root = np.cbrt(weights / scale) if scale else weights.copy()
    if directed:
        degree = a.sum(axis=0) + a.sum(axis=1)
        reciprocal = (a * a.T).sum(axis=1)
        denominator = 2 * (degree * (degree - 1) - 2 * reciprocal)
        binary = a + a.T
        weighted = root + root.T
    else:
        degree = a.sum(axis=1)
        denominator = degree * (degree - 1)
        binary, weighted = a, root
    triangle_b = np.einsum('ij,ji->i', binary @ binary, binary)
    triangle_w = np.einsum('ij,ji->i', weighted @ weighted, weighted)
    return tuple(np.divide(x, denominator, out=np.zeros(len(a)), where=denominator > 0)
                 for x in (triangle_b, triangle_w))


def _eigenvector(weights, directed):
    n = len(weights)
    if not np.any(weights):
        return np.full(n, np.nan), 'undefined_edgeless'
    values, vectors = (np.linalg.eig(weights.T) if directed else np.linalg.eigh(weights))
    largest = int(np.argmax(values.real))
    eigenvalue = values[largest]
    tolerance = max(1e-14, abs(eigenvalue) * 1e-9)
    if abs(eigenvalue) <= tolerance or np.count_nonzero(np.abs(values - eigenvalue) <= tolerance) != 1:
        return np.full(n, np.nan), 'undefined_nonunique_or_zero_eigenvalue'
    vector = vectors[:, largest]
    if np.max(np.abs(vector.imag)) > 1e-8:
        return np.full(n, np.nan), 'undefined_complex_eigenvector'
    vector = vector.real
    if vector.sum() < 0:
        vector = -vector
    if vector.min() < -1e-8:
        return np.full(n, np.nan), 'undefined_nonpositive_eigenvector'
    vector = np.maximum(vector, 0)
    return vector / np.linalg.norm(vector), 'ok'


def compute_network_metrics(
    connectivity: np.ndarray,
    *,
    channels: Sequence | None = None,
    directed: bool = False,
    threshold: float = 0.0,
    negative_policy: str = 'raise',
    nan_policy: str = 'raise',
) -> NetworkMetricsResult:
    """Compute degree, strength, centrality, clustering and integration metrics.

    Parameters
    ----------
    connectivity
        Square real matrix. For directed graphs, row i is the source and column j
        the target. Transpose TAD peak cross-correlation matrices before passing
        them here. Undirected matrices must be symmetric (absolute tolerance 1e-12).
    channels
        Unique node identifiers in matrix order; defaults to integer indices.
    directed
        Explicit graph orientation; asymmetry is never silently symmetrized.
    threshold
        Keep strengths strictly greater than this nonnegative value. Default 0
        retains all positive estimated associations, NOT statistically tested edges.
    negative_policy
        'raise' (default), 'positive' (discard negative edges), or 'absolute'.
        The raw input is never modified. Signed input strengths are also returned.
    nan_policy
        'raise' (default) or 'zero' (unavailable entries become absent edges).
        Infinities always raise. With 'zero', unavailable nodes remain in the graph.

    Notes
    -----
    Path lengths are 1 / retained strength; unreachable pairs contribute zero to
    efficiency and normalized harmonic centrality. The latter is reported both
    inward and outward for directed graphs. Betweenness is weighted, normalized,
    and excludes endpoints. Degree centrality uses n-1 possible neighbours.
    Directed degree/strength are totals (in + out), with in/out also returned.

    Weighted clustering is Onnela (undirected) / Fagiolo (directed), normalized
    by this graph's largest retained weight, as in NetworkX. Other strength/path
    metrics are NOT rescaled. Efficiency can exceed 1 for strengths above 1.
    Eigenvector centrality uses the incoming/left eigenvector, L2 normalized;
    ambiguous or edgeless cases are NaN with an explicit status in ``info``.
    Mean reachable path length excludes unreachable pairs; component sizes and
    reachable fraction must be considered alongside it. No communities, causal
    claims, edge significance, or automatic density matching are inferred.
    """
    original = np.asarray(connectivity)
    if np.iscomplexobj(original):
        raise ValueError('connectivity must be real')
    w = np.array(original, dtype=float, copy=True)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or not len(w):
        raise ValueError('connectivity must be a nonempty square matrix')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    if negative_policy not in {'raise', 'positive', 'absolute'}:
        raise ValueError('negative_policy must be raise, positive or absolute')
    if nan_policy not in {'raise', 'zero'}:
        raise ValueError('nan_policy must be raise or zero')
    if np.isinf(w).any():
        raise ValueError('infinite connectivity values are not supported')
    n = len(w)
    labels = list(range(n)) if channels is None else list(channels)
    if len(labels) != n or len(set(labels)) != n:
        raise ValueError('channels must contain one unique identifier per matrix row')
    np.fill_diagonal(w, 0)
    if not directed and not np.allclose(w, w.T, atol=1e-12, rtol=0, equal_nan=True):
        raise ValueError('undirected connectivity must be symmetric')
    missing = int(np.isnan(w).sum())
    if missing and nan_policy == 'raise':
        raise ValueError('NaN connectivity values require an explicit nan_policy')
    w[np.isnan(w)] = 0
    positive = np.maximum(w, 0)
    negative = np.maximum(-w, 0)
    if negative.any() and negative_policy == 'raise':
        raise ValueError('negative connectivity requires an explicit negative_policy')
    w = np.abs(w) if negative_policy == 'absolute' else positive.copy()
    w[w <= threshold] = 0
    if not directed:
        w = (w + w.T) / 2
    a = w > 0
    lengths = np.divide(1.0, w, out=np.zeros_like(w), where=a)
    if not np.isfinite(lengths).all():
        raise ValueError('weights are too small for finite inverse-strength distances')
    graph = nx.from_numpy_array(w, create_using=nx.DiGraph if directed else nx.Graph)
    for i, j, data in graph.edges(data=True):
        data['distance'] = lengths[i, j]
    distances = shortest_path(lengths, directed=directed, method='FW')
    offdiag = ~np.eye(n, dtype=bool)
    reachable = offdiag & np.isfinite(distances)
    inverse = np.divide(1.0, distances, out=np.zeros_like(w), where=reachable)
    denom = max(n - 1, 1)
    binary_cluster, weighted_cluster = _clustering(w, directed)
    eigenvector, eigen_status = _eigenvector(w, directed)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight='distance')
    degree_out, degree_in = a.sum(axis=1), a.sum(axis=0)
    strength_out, strength_in = w.sum(axis=1), w.sum(axis=0)
    degree = degree_out + degree_in if directed else degree_out
    strength = strength_out + strength_in if directed else strength_out
    node = {
        'degree': degree, 'degree_centrality': degree / denom,
        'strength': strength, 'betweenness': np.array([betweenness[i] for i in range(n)]),
        'eigenvector': eigenvector, 'clustering': binary_cluster,
        'weighted_clustering': weighted_cluster,
        'harmonic_in': inverse.sum(axis=0) / denom,
        'harmonic_out': inverse.sum(axis=1) / denom,
        'positive_input_strength_out': positive.sum(axis=1),
        'negative_input_strength_out': negative.sum(axis=1),
    }
    if directed:
        node.update(in_degree=degree_in, out_degree=degree_out,
                    in_degree_centrality=degree_in / denom, out_degree_centrality=degree_out / denom,
                    in_strength=strength_in, out_strength=strength_out)
    weak = list(nx.weakly_connected_components(graph) if directed else nx.connected_components(graph))
    strong = list(nx.strongly_connected_components(graph)) if directed else weak
    edges = graph.number_of_edges()
    global_metrics = {
        'n_nodes': n, 'n_edges': edges, 'density': float(nx.density(graph)),
        'n_isolates': int(np.sum(degree == 0)),
        'mean_degree': float(degree.mean()), 'mean_strength': float(strength.mean()),
        'mean_clustering': float(binary_cluster.mean()),
        'mean_weighted_clustering': float(weighted_cluster.mean()),
        'global_efficiency': float(inverse.sum() / (n * denom)),
        'reachable_pair_fraction': float(reachable.sum() / (n * denom)),
        'mean_reachable_path_length': float(distances[reachable].mean()) if reachable.any() else float('nan'),
        'n_weak_components': len(weak), 'n_strong_components': len(strong),
        'largest_weak_component_fraction': max(map(len, weak)) / n,
        'largest_strong_component_fraction': max(map(len, strong)) / n,
        'reciprocity': float((a & a.T).sum() / a.sum()) if directed and edges else float('nan'),
    }
    return NetworkMetricsResult(labels, w, node, global_metrics, {
        'directed': directed, 'threshold': threshold, 'negative_policy': negative_policy,
        'nan_policy': nan_policy, 'unavailable_entries': missing,
        'matrix_convention': 'row source -> column target', 'distance': '1 / strength',
        'weighted_clustering_scale': float(w.max(initial=0)),
        'eigenvector_status': eigen_status, 'networkx_version': nx.__version__,
    })
