Metrics
=======

Scalar Metrics
--------------

.. automodule:: tad.metrics.scalar
   :members:
   :undoc-members:
   :show-inheritance:

Firing Rate
-----------

.. automodule:: tad.metrics.rates
   :members:
   :undoc-members:
   :show-inheritance:

Avalanches
----------

.. automodule:: tad.metrics.avalanches
   :members:
   :undoc-members:
   :show-inheritance:

Bursts
------

.. automodule:: tad.metrics.burst
   :members:
   :undoc-members:
   :show-inheritance:

ISI Analysis
------------

.. automodule:: tad.metrics.isi
   :members:
   :undoc-members:
   :show-inheritance:

Synchrony
---------

.. automodule:: tad.metrics.synchrony
   :members:
   :undoc-members:
   :show-inheritance:

Cross-Correlation
-----------------

.. automodule:: tad.metrics.crosscorrelation
   :members:
   :undoc-members:
   :show-inheritance:

Power Law Fitting
-----------------

.. automodule:: tad.metrics.powerlaw
   :members:
   :undoc-members:
   :show-inheritance:

Evoked Responses
----------------

.. automodule:: tad.metrics.evoked
   :members:
   :undoc-members:
   :show-inheritance:

PSTH (Post-Stimulus Time Histogram)
-----------------------------------

.. automodule:: tad.metrics.psth
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: tad.metrics.utils
   :members:
   :undoc-members:
   :show-inheritance:


Network Metrics
---------------

.. automodule:: tad.metrics.network
   :members:
   :undoc-members:
   :show-inheritance:

Example::

    from tad.metrics import compute_network_metrics

    result = compute_network_metrics(matrix, channels=channel_ids,
                                     negative_policy="positive")
    strength = result.node_metrics["strength"]
    betweenness = result.node_metrics["betweenness"]
    efficiency = result.global_metrics["global_efficiency"]

For directed matrices, set ``directed=True`` and supply rows as sources and columns
as targets. TAD's peak cross-correlation result uses the opposite orientation, so
pass ``correlation.T``. Choose the negative-weight and missing-value policies
explicitly; the input matrix is never mutated. See the function documentation for
weight, distance, clustering-normalization, and disconnected-graph conventions.
