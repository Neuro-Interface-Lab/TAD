"""
processing_history.py

Lightweight provenance tracking for data processing steps.

Design goals
------------
- JSON-first persistence (portable, diffable)
- Minimal intrusion into existing processing code
- Captures ordered operations (B) and key state snapshots (C)
- Lightweight dataset metadata (A-lite): name, sampling rate, channels, file stats
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _canonical_dumps(obj: Any) -> str:
    """Deterministic JSON string used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_json_value(x: Any) -> Any:
    """
    Convert common non-JSON objects to JSON-friendly representations.

    Notes
    -----
    keep this conservative: it should not accidentally serialize large arrays.
    """
    # Basic scalars
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # Simple containers
    if isinstance(x, (list, tuple)):
        return [_safe_json_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_json_value(v) for k, v in x.items()}

    # Numpy types / arrays (without importing numpy as hard dependency here)
    tname = type(x).__name__
    mod = getattr(type(x), "__module__", "")
    if mod.startswith("numpy"):
        # scalar -> python scalar
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                pass

        # array -> small list only; otherwise require caller to pre-encode
        if hasattr(x, "shape") and hasattr(x, "size"):
            size = int(getattr(x, "size", 0))
            if size <= 256 and hasattr(x, "tolist"):
                return x.tolist()
            return {"__ndarray__": True, "note": f"array too large to inline (size={size})"}

    # Fallback
    return str(x)


def encode_bool_mask(mask_bytes: bytes) -> str:
    """
    Base64 encode a boolean mask serialized as raw bytes.

    Parameters
    ----------
    mask_bytes : bytes
        Raw bytes representation of a boolean mask (e.g., numpy packbits output).

    Returns
    -------
    str
        Base64-encoded string.
    """
    return base64.b64encode(mask_bytes).decode("ascii")


def decode_bool_mask(encoded: str) -> bytes:
    """
    Decode a base64-encoded boolean mask payload.

    Parameters
    ----------
    encoded : str
        Base64-encoded mask string.

    Returns
    -------
    bytes
        Decoded bytes.
    """
    return base64.b64decode(encoded.encode("ascii"))


def tracked_operation(
    name: Optional[str] = None,
    *,
    track: bool = True,
    include_result_artifacts: Optional[Callable[[Any], Dict[str, Any]]] = None,
) -> Callable:
    """
    Decorator to record a method call into provenance.

    It supports two targets (if present on `self`):
      - self.processing_history : ProcessingHistory-like object with snapshot_state/state_hash/record
      - self.history : list[dict] (lightweight generic log)

    Parameters
    ----------
    name : str, optional
        Operation name. Defaults to function name.
    track : bool, default=True
        Whether to track this operation.
    include_result_artifacts : callable, optional
        Function called with the method return value; must return JSON-friendly dict
        to attach as `artifacts`.
    """
    def _decorator(func: Callable) -> Callable:
        op_name = name or func.__name__

        def _wrapped(self, *args, **kwargs):
            if not track:
                return func(self, *args, **kwargs)

            # --- resolve history targets (robustly) ---
            ph = getattr(self, "processing_history", None)
            lightweight = getattr(self, "history", None)

            has_ph = (
                ph is not None
                and hasattr(ph, "snapshot_state")
                and hasattr(ph, "state_hash")
                and hasattr(ph, "record")
            )
            has_lightweight = isinstance(lightweight, list)

            # If nothing to log to, behave like a no-op decorator
            if (not has_ph) and (not has_lightweight):
                return func(self, *args, **kwargs)

            # --- before snapshot/hash (ProcessingHistory only) ---
            before_hash = None
            if has_ph:
                try:
                    before_snapshot = ph.snapshot_state()
                    before_hash = ph.state_hash(before_snapshot)
                except Exception:
                    before_hash = None

            # run operation
            result = func(self, *args, **kwargs)

            # --- after snapshot/hash (ProcessingHistory only) ---
            after_hash = None
            after_snapshot = None
            if has_ph:
                try:
                    after_snapshot = ph.snapshot_state()
                    after_hash = ph.state_hash(after_snapshot)
                except Exception:
                    after_hash = None
                    after_snapshot = None

            # conservative params capture
            params = _safe_json_value({"args": list(args), "kwargs": dict(kwargs)})

            artifacts: Dict[str, Any] = {}
            if include_result_artifacts is not None:
                try:
                    artifacts = include_result_artifacts(result) or {}
                except Exception:
                    artifacts = {"note": "artifact extraction failed"}
            artifacts = _safe_json_value(artifacts)

            # --- record to ProcessingHistory (rich) ---
            if has_ph:
                # Optional summary derived from snapshot if the getter provides these keys
                summary: Dict[str, Any] = {}
                if isinstance(after_snapshot, dict):
                    if "mask_n_kept" in after_snapshot:
                        summary["mask_n_kept"] = after_snapshot.get("mask_n_kept")
                    if "excluded_intervals_n" in after_snapshot:
                        summary["excluded_intervals_n"] = after_snapshot.get("excluded_intervals_n")

                try:
                    ph.record(
                        op_name,
                        params=params,
                        state_before=before_hash,
                        state_after=after_hash,
                        summary=_safe_json_value(summary),
                        artifacts=artifacts,
                    )
                except Exception:
                    # Don't break the pipeline because provenance failed
                    pass

            # --- record to lightweight list[dict] ---
            if has_lightweight:
                try:
                    lightweight.append(
                        _safe_json_value(
                            {
                                "name": op_name,
                                "timestamp_utc": _utc_now_iso(),
                                "params": params,
                                "state_before": before_hash,
                                "state_after": after_hash,
                                "artifacts": artifacts,
                            }
                        )
                    )
                except Exception:
                    pass

            return result

        _wrapped.__name__ = func.__name__
        _wrapped.__doc__ = func.__doc__
        _wrapped.__qualname__ = func.__qualname__
        return _wrapped

    return _decorator

@dataclass
class DatasetInfo:
    """
    Lightweight dataset identity/metadata (A-lite).

    Attributes
    ----------
    fname : str
        Path (as provided) to the source file.
    basename : str
        Base filename.
    sampling_frequency : float, optional
        Sampling frequency in Hz.
    stream_id : int, optional
        Stream identifier used by the reader (if applicable).
    channel_ids : list, optional
        Channel identifiers.
    electrode_labels : list, optional
        Electrode labels (if present).
    file_size_bytes : int, optional
        File size in bytes (if path exists).
    file_mtime_utc : str, optional
        File modification time in UTC ISO string.
    """

    fname: str
    basename: str
    sampling_frequency: Optional[float] = None
    stream_id: Optional[int] = None
    channel_ids: Optional[List[Any]] = None
    electrode_labels: Optional[List[Any]] = None
    file_size_bytes: Optional[int] = None
    file_mtime_utc: Optional[str] = None

    @classmethod
    def from_path(
        cls,
        fname: str,
        sampling_frequency: Optional[float] = None,
        stream_id: Optional[int] = None,
        channel_ids: Optional[List[Any]] = None,
        electrode_labels: Optional[List[Any]] = None,
    ) -> "DatasetInfo":
        basename = os.path.basename(fname)
        size = None
        mtime = None
        if os.path.exists(fname):
            st = os.stat(fname)
            size = int(st.st_size)
            mtime = _dt.datetime.fromtimestamp(st.st_mtime, tz=_dt.timezone.utc).isoformat()

        return cls(
            fname=fname,
            basename=basename,
            sampling_frequency=sampling_frequency,
            stream_id=stream_id,
            channel_ids=channel_ids,
            electrode_labels=electrode_labels,
            file_size_bytes=size,
            file_mtime_utc=mtime,
        )

    def to_dict(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class OperationRecord:
    """
    A single processing operation record (B).

    Attributes
    ----------
    name : str
        Operation name (e.g., "apply_filter").
    timestamp_utc : str
        UTC timestamp ISO.
    params : dict
        JSON-friendly parameters.
    state_before : str, optional
        Hash of state snapshot before operation.
    state_after : str, optional
        Hash of state snapshot after operation.
    summary : dict, optional
        Small human-friendly summary (counts, etc.).
    artifacts : dict, optional
        Any extra “result metadata” (e.g., raster t-range, n_spikes).
    """

    name: str
    timestamp_utc: str
    params: JsonDict = field(default_factory=dict)
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    summary: JsonDict = field(default_factory=dict)
    artifacts: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return dataclasses.asdict(self)


@dataclass
class ProcessingHistory:
    """
    Provenance container for a processing session.

    Parameters
    ----------
    dataset : DatasetInfo
        Lightweight dataset metadata.
    session_id : str, optional
        Stable id for this history instance. If None, a hash is generated.
    """

    dataset: DatasetInfo
    operations: List[OperationRecord] = field(default_factory=list)
    created_utc: str = field(default_factory=_utc_now_iso)
    session_id: str = field(default_factory=lambda: _sha256_text(_utc_now_iso()))

    # Not persisted: function to snapshot state from a live object
    _state_getter: Optional[Callable[[], JsonDict]] = field(default=None, repr=False, compare=False)

    def set_state_getter(self, getter: Callable[[], JsonDict]) -> None:
        """Attach a callback that returns a JSON-friendly state snapshot."""
        self._state_getter = getter

    def snapshot_state(self) -> JsonDict:
        """Return a state snapshot using the attached getter, or empty dict."""
        if self._state_getter is None:
            return {}
        snap = self._state_getter()
        return _safe_json_value(snap)

    def state_hash(self, snapshot: Optional[JsonDict] = None) -> str:
        """
        Compute a stable hash for a given snapshot (or current snapshot if None).
        """
        if snapshot is None:
            snapshot = self.snapshot_state()
        return _sha256_text(_canonical_dumps(snapshot))

    def record(
        self,
        name: str,
        params: Optional[JsonDict] = None,
        *,
        state_before: Optional[str] = None,
        state_after: Optional[str] = None,
        summary: Optional[JsonDict] = None,
        artifacts: Optional[JsonDict] = None,
    ) -> None:
        """
        Append an operation record.

        Parameters
        ----------
        name : str
            Operation name.
        params : dict, optional
            Operation parameters (JSON-friendly).
        state_before : str, optional
            State hash before op.
        state_after : str, optional
            State hash after op.
        summary : dict, optional
            Human-friendly summary.
        artifacts : dict, optional
            Output-related metadata.
        """
        rec = OperationRecord(
            name=name,
            timestamp_utc=_utc_now_iso(),
            params=_safe_json_value(params or {}),
            state_before=state_before,
            state_after=state_after,
            summary=_safe_json_value(summary or {}),
            artifacts=_safe_json_value(artifacts or {}),
        )
        self.operations.append(rec)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-friendly dict."""
        return {
            "schema": "tad.processing_history.v1",
            "created_utc": self.created_utc,
            "session_id": self.session_id,
            "dataset": self.dataset.to_dict(),
            "operations": [op.to_dict() for op in self.operations],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: str, indent: int = 2) -> None:
        """Write history JSON to disk."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, d: JsonDict) -> "ProcessingHistory":
        dataset = DatasetInfo(**d["dataset"])
        hist = cls(dataset=dataset, created_utc=d.get("created_utc", _utc_now_iso()), session_id=d.get("session_id", ""))
        for op in d.get("operations", []):
            hist.operations.append(OperationRecord(**op))
        return hist

    @classmethod
    def load_json(cls, path: str) -> "ProcessingHistory":
        """Load history JSON from disk."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)