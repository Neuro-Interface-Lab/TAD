from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from tad.metrics.avalanches import AvalancheResult


@dataclass(frozen=True)
class PowerLawFitResult:
    """
    Power-law fit result for discrete avalanche distributions.

    Parameters
    ----------
    variable
        "size" or "lifetime".
    a
        Prefactor in P(x) = a * x^b (in probability space).
    b
        Exponent in P(x) = a * x^b.
    rmse
        Root mean squared error between empirical P(x) and fitted P_hat(x),
        computed over the retained bins (after exclusions).
    x_all
        All support values (sorted unique x) from the empirical PMF.
    p_all
        Empirical PMF values corresponding to x_all.
    keep_mask
        Boolean mask over x_all/p_all indicating which points were used for fit/RMSE.
    x_used
        x values used for fit.
    p_used
        empirical P(x) used for fit.
    p_hat_used
        fitted P_hat(x) evaluated at x_used.
    exclude_unit
        Whether x=1 was excluded.
    tail_fraction
        Tail cutoff fraction relative to max(P). Points with P < tail_fraction * max(P)
        were excluded.
    """
    variable: Literal["size", "lifetime"]
    a: float
    b: float
    rmse: float
    x_all: np.ndarray
    p_all: np.ndarray
    keep_mask: np.ndarray
    x_used: np.ndarray
    p_used: np.ndarray
    p_hat_used: np.ndarray
    exclude_unit: bool
    tail_fraction: float


def _pmf_from_samples(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute discrete PMF from positive integer samples.

    Parameters
    ----------
    samples
        1D array of integers (>=1).

    Returns
    -------
    x
        Sorted unique support values.
    p
        Probability mass P(x) on that support.
    """
    s = np.asarray(samples, dtype=np.int64).ravel()
    s = s[s >= 1]
    if s.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    x, c = np.unique(s, return_counts=True)
    p = c.astype(np.float64) / float(c.sum())
    return x.astype(np.float64), p


def fit_avalanche_powerlaw(
    aval: AvalancheResult,
    *,
    variable: Literal["size", "lifetime"] = "size",
    exclude_unit: bool = True,
    tail_fraction: float = 0.01,
    min_points: int = 5,
) -> PowerLawFitResult:
    """
    Fit a power-law model P(x) = a x^b to an avalanche distribution and compute RMSE.

    This implements the *paper-style* protocol used in your references:
    - Build a discrete histogram / PMF of sizes or lifetimes
    - Exclude unit events (x=1) (first bin)
    - Exclude tail bins whose probability is < (tail_fraction * max_probability)
      (paper uses 1% of the maximum => tail_fraction=0.01)
    - Fit P(x) = a x^b
    - Compute RMSE between empirical P(x) and fitted P_hat(x) on retained points

    Parameters
    ----------
    aval
        AvalancheResult containing extracted avalanche sizes/lifetimes.
    variable
        Choose which distribution to fit: "size" or "lifetime".
    exclude_unit
        If True, exclude x=1.
    tail_fraction
        Tail probability cutoff relative to max(P). Default 0.01.
    min_points
        Minimum number of retained points required to fit. If fewer, returns NaNs.

    Returns
    -------
    PowerLawFitResult
        Fit parameters and diagnostics.

    Notes
    -----
    The fit is performed by linear least squares on log space:
        log P = log a + b log x
    Then a is recovered by exponentiation.

    RMSE is computed in probability space:
        RMSE = sqrt(mean((P - P_hat)^2))
    over the retained points, matching the “goodness-of-fit via RMSE” style described.
    """
    if tail_fraction <= 0.0 or not np.isfinite(tail_fraction):
        raise ValueError(f"tail_fraction must be positive and finite, got {tail_fraction!r}.")
    if min_points < 3:
        raise ValueError("min_points should be >= 3.")

    if variable == "size":
        samples = aval.sizes
    elif variable == "lifetime":
        samples = aval.lifetimes
    else:
        raise ValueError(f"variable must be 'size' or 'lifetime', got {variable!r}.")

    x_all, p_all = _pmf_from_samples(samples)

    # Default keep mask: all points with positive probability
    keep = (p_all > 0.0)

    # Exclude unit avalanches (first bin)
    if exclude_unit:
        keep &= (x_all != 1.0)

    # Tail trimming: remove bins with P < tail_fraction * max(P) among currently kept bins
    if np.any(keep):
        pmax = float(np.max(p_all[keep]))
        keep &= (p_all >= tail_fraction * pmax)

    x_used = x_all[keep]
    p_used = p_all[keep]

    # Not enough points: return NaNs but keep artifacts for inspection
    if x_used.size < min_points:
        nan = float("nan")
        return PowerLawFitResult(
            variable=variable,
            a=nan,
            b=nan,
            rmse=nan,
            x_all=x_all,
            p_all=p_all,
            keep_mask=keep,
            x_used=x_used,
            p_used=p_used,
            p_hat_used=np.asarray([], dtype=np.float64),
            exclude_unit=exclude_unit,
            tail_fraction=float(tail_fraction),
        )

    # Fit in log space: log(P) = log(a) + b log(x)
    lx = np.log(x_used)
    lp = np.log(p_used)

    # Linear regression (least squares)
    # lp ≈ intercept + b * lx
    b, intercept = np.polyfit(lx, lp, deg=1)
    a = float(np.exp(intercept))

    p_hat = a * (x_used ** b)

    rmse = float(np.sqrt(np.mean((p_used - p_hat) ** 2)))

    return PowerLawFitResult(
        variable=variable,
        a=a,
        b=float(b),
        rmse=rmse,
        x_all=x_all,
        p_all=p_all,
        keep_mask=keep,
        x_used=x_used,
        p_used=p_used,
        p_hat_used=p_hat,
        exclude_unit=exclude_unit,
        tail_fraction=float(tail_fraction),
    )
