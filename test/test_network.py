import networkx as nx
import numpy as np
import pytest

from tad.metrics.network import compute_network_metrics


def test_path_metrics_and_input_unchanged():
    w = np.array([[8., 2, 0], [2, 8, 2], [0, 2, 8]])
    original = w.copy()
    r = compute_network_metrics(w, channels=['a', 'b', 'c'])
    np.testing.assert_array_equal(w, original)
    np.testing.assert_array_equal(r.node_metrics['degree'], [1, 2, 1])
    np.testing.assert_allclose(r.node_metrics['strength'], [2, 4, 2])
    np.testing.assert_allclose(r.node_metrics['betweenness'], [0, 1, 0])
    np.testing.assert_allclose(r.node_metrics['eigenvector'], [.5, np.sqrt(.5), .5])
    assert r.global_metrics['global_efficiency'] == pytest.approx(5 / 3)
    assert r.global_metrics['density'] == pytest.approx(2 / 3)
    assert not np.diag(r.adjacency).any()


def test_directed_path_orientation_and_unreachable_pairs():
    r = compute_network_metrics(np.array([[0., 1, 0], [0, 0, 1], [0, 0, 0]]), directed=True)
    np.testing.assert_array_equal(r.node_metrics['out_degree'], [1, 1, 0])
    np.testing.assert_array_equal(r.node_metrics['in_degree'], [0, 1, 1])
    np.testing.assert_allclose(r.node_metrics['harmonic_out'], [.75, .5, 0])
    np.testing.assert_allclose(r.node_metrics['harmonic_in'], [0, .5, .75])
    np.testing.assert_allclose(r.node_metrics['betweenness'], [0, .5, 0])
    assert r.global_metrics['global_efficiency'] == pytest.approx(2.5 / 6)
    assert r.global_metrics['n_strong_components'] == 3
    assert np.isnan(r.node_metrics['eigenvector']).all()


@pytest.mark.parametrize('directed', [False, True])
def test_weighted_clustering_matches_networkx(directed):
    rng = np.random.default_rng(15)
    w = rng.uniform(0, 3, (8, 8))
    w[w < 1] = 0
    if not directed:
        w = (w + w.T) / 2
    np.fill_diagonal(w, 0)
    r = compute_network_metrics(w, directed=directed)
    g = nx.from_numpy_array(w, create_using=nx.DiGraph if directed else nx.Graph)
    np.testing.assert_allclose(r.node_metrics['clustering'], list(nx.clustering(g).values()))
    np.testing.assert_allclose(r.node_metrics['weighted_clustering'], list(nx.clustering(g, weight='weight').values()))


def test_signed_nan_policies_and_threshold():
    w = np.array([[0., -2, np.nan], [-2, 0, .2], [np.nan, .2, 0]])
    with pytest.raises(ValueError):
        compute_network_metrics(w)
    with pytest.raises(ValueError):
        compute_network_metrics(w, nan_policy='zero')
    r = compute_network_metrics(w, nan_policy='zero', negative_policy='absolute', threshold=.2)
    np.testing.assert_array_equal(r.node_metrics['degree'], [1, 1, 0])
    assert r.info['unavailable_entries'] == 2
    assert r.global_metrics['largest_weak_component_fraction'] == pytest.approx(2 / 3)
    assert r.global_metrics['global_efficiency'] == pytest.approx(2 / 3)
    p = compute_network_metrics(w, nan_policy='zero', negative_policy='positive')
    np.testing.assert_allclose(p.node_metrics['negative_input_strength_out'], [2, 2, 0])


@pytest.mark.parametrize('n', [1, 4])
def test_edgeless_and_single_node(n):
    r = compute_network_metrics(np.zeros((n, n)))
    assert r.global_metrics['global_efficiency'] == 0
    assert r.global_metrics['density'] == 0
    assert r.global_metrics['n_isolates'] == n
    assert np.isnan(r.global_metrics['mean_reachable_path_length'])
    assert np.isnan(r.node_metrics['eigenvector']).all()


def test_nonunique_eigenvector_is_flagged():
    r = compute_network_metrics(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
    assert np.isnan(r.node_metrics['eigenvector']).all()
    assert 'nonunique' in r.info['eigenvector_status']


@pytest.mark.parametrize('matrix, kwargs', [
    (np.zeros((2, 3)), {}), (np.zeros((0, 0)), {}),
    (np.array([[0., 1], [0, 0]]), {}),
    (np.array([[0., np.inf], [np.inf, 0]]), {}),
    (np.zeros((2, 2)), {'channels': ['a', 'a']}),
    (np.zeros((2, 2)), {'threshold': -1}),
    (np.zeros((2, 2)), {'negative_policy': 'guess'}),
])
def test_invalid_inputs(matrix, kwargs):
    with pytest.raises(ValueError):
        compute_network_metrics(matrix, **kwargs)
