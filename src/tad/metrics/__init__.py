from .scalar import (
    spike_count,
    firing_rate,
    mean_firing_rate_across_channels,
    mean_inter_event_interval,
    percent_random_spiking,
)
from .avalanches import extract_avalanches, AvalancheResult
from .rates import firing_rate_curve, FiringRateCurveResult
from .isi import isi, isih, ISIResult
from .powerlaw import fit_avalanche_powerlaw, PowerLawFitResult
from .synchrony import PearsonSynchronyResult, pearson_corr_firing_rate
from .burst import Burst, BurstChannelResult, BurstDetectionResult, detect_bursts
from .psth import compute_psth, PSTHResult
from .evoked import EvokedPeakResult, evoked_peak_metrics, response_probability
from .spectral_firingrates import (
    FiringRatePowerSpectrumResult,
    FiringRateSpectrogramResult,
    FiringRateAutocorrelationResult,
    power_spectrum_firing_rate,
    spectrogram_firing_rate,
    autocorrelation_firing_rate,
)
from .hilbert_firingrates import (
    HilbertFiringRateResult,
    PhaseLockingValueResult,
    KuramotoOrderResult,
    hilbert_firing_rate,
    phase_locking_value_firing_rate,
    kuramoto_order_firing_rate,
)

# from .utils import _select_channels, _infer_window, pooled_spike_times

__all__ = [
    "spike_count",
    "firing_rate",
    "mean_firing_rate_across_channels",
    "mean_inter_event_interval",
    "isi",
    "isih",
    "ISIResult" "percent_random_spiking",
    "AvalancheResult",
    "extract_avalanches",
    "firing_rate_curve",
    "FiringRateCurveResult",
    "fit_avalanche_powerlaw",
    "PowerLawFitResult",
    "PearsonSynchronyResult",
    "pearson_corr_firing_rate",
    "Burst",
    "BurstChannelResult",
    "BurstDetectionResult",
    "detect_bursts",
    "compute_psth",
    "PSTHResult",
    "EvokedPeakResult",
    "evoked_peak_metrics",
    "response_probability",
    "FiringRatePowerSpectrumResult",
    "FiringRateSpectrogramResult",
    "FiringRateAutocorrelationResult",
    "power_spectrum_firing_rate",
    "spectrogram_firing_rate",
    "autocorrelation_firing_rate",
    "HilbertFiringRateResult",
    "PhaseLockingValueResult",
    "KuramotoOrderResult",
    "hilbert_firing_rate",
    "phase_locking_value_firing_rate",
    "kuramoto_order_firing_rate",
]
