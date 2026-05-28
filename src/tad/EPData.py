from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .raster import Raster


def get_raster_from_csv(
    fname: Union[str, Path],
    *,
    delimiter: str = ",",
    skiprows: int = 0,
    comments: str = "#",
    channels: Optional[Iterable[int]] = None,
    dtype: np.dtype = np.dtype(np.float64),
    channel_dtype: type = int,
    allow_new_channels: bool = True,
) -> Raster:
    """
    Create a Raster from a CSV-like text file.

    The input file is expected to contain at least two numerical columns:

        column 0: timestamps
        column 1: channel identifiers

    Example
    -------
    0.0001,4
    0.0001,6
    0.00035,23

    Parameters
    ----------
    fname:
        Path to the CSV file.

    delimiter:
        Column delimiter. Default is ",".

    skiprows:
        Number of initial rows to skip. Use this if the file has a header.

    comments:
        Character used to indicate comments in the file.

    channels:
        Optional iterable of channels to pre-create in the Raster.
        If None, channels are inferred from the file.

    dtype:
        NumPy dtype used for timestamps in the Raster.

    channel_dtype:
        Type used to cast channel identifiers. Default is int.

    allow_new_channels:
        Passed to Raster.empty. If True, missing channels can be created
        during insertion.

    Returns
    -------
    Raster
        A Raster containing all timestamps grouped by channel.

    Raises
    ------
    ValueError
        If the file does not contain at least two columns, if timestamps are
        non-finite, or if channel identifiers cannot be safely converted.
    """
    path = Path(fname)

    try:
        data = np.loadtxt(
            path,
            delimiter=delimiter,
            skiprows=skiprows,
            comments=comments,
            ndmin=2,
        )
    except OSError as exc:
        raise OSError(f"Could not open CSV file: {path}") from exc
    except ValueError as exc:
        raise ValueError(
            f"Could not parse CSV file {path}. "
            "Expected a numerical file with timestamps in column 0 "
            "and channels in column 1."
        ) from exc

    if data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}, got {data.shape[1]}.")

    timestamps = np.asarray(data[:, 0], dtype=dtype)
    raw_channels = data[:, 1]

    if not np.all(np.isfinite(timestamps)):
        raise ValueError("All timestamps must be finite.")

    if not np.all(np.isfinite(raw_channels)):
        raise ValueError("All channel identifiers must be finite.")

    # Default case: channels are numerical IDs such as 1, 2, 3, ...
    # We explicitly check that they are integer-valued before casting.
    if channel_dtype is int:
        if not np.all(raw_channels == np.floor(raw_channels)):
            raise ValueError("Channel identifiers must be integer-valued when " "channel_dtype=int.")

    channel_ids = raw_channels.astype(channel_dtype)

    if channels is None:
        channels = np.unique(channel_ids)

    raster = Raster.empty(
        channels=channels,
        dtype=dtype,
        allow_new_channels=allow_new_channels,
    )

    for ch in np.unique(channel_ids):
        mask = channel_ids == ch
        raster.insert_timestamparray(
            channel_dtype(ch),
            timestamps[mask],
            assume_sorted=False,
        )

    return raster
