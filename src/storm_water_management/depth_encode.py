"""Encoding of depression depth as a single uint8 band.

Depression depths span roughly 0 to 30+ m (large lakes appear as deep
depressions since the 50 m elevation model shows water surfaces, not
bathymetry). To stay useful for visualisation at low depths in one
uint8 band, the values are non-linearly scaled with either a sqrt or a
log1p transform, saturating at ``depth_max``.
"""

import math

import numpy as np

ENCODINGS = ("sqrt", "log")
DEFAULT_DEPTH_MAX = 30.0
DEFAULT_ENCODING = "sqrt"


def encode_depth(
    depth: np.ndarray,
    encoding: str = DEFAULT_ENCODING,
    depth_max: float = DEFAULT_DEPTH_MAX,
) -> np.ndarray:
    """Encode depression depth (m) into a uint8 band.

    Args:
        depth: depression depth in meters, NaN where no data.
        encoding: ``"sqrt"`` or ``"log"``
            (``255 * sqrt(min(depth, max) / max)`` or
            ``255 * log1p(min(depth, max)) / log1p(max)``).
        depth_max: saturation depth in meters.

    Returns:
        uint8 array, 0 where there is no depression, NaN input
        becomes 0.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding {encoding!r}, use one of {ENCODINGS}")
    d = np.nan_to_num(np.asarray(depth, dtype=np.float64), nan=0.0)
    d = np.clip(d, 0.0, depth_max)
    if encoding == "sqrt":
        scaled = np.sqrt(d / depth_max)
    else:
        scaled = np.log1p(d) / math.log1p(depth_max)
    return np.round(scaled * 255.0).astype(np.uint8)


def decode_depth(
    band: np.ndarray,
    encoding: str = DEFAULT_ENCODING,
    depth_max: float = DEFAULT_DEPTH_MAX,
) -> np.ndarray:
    """Decode a uint8 band back to depression depth (m).

    Args:
        band: uint8 band from :func:`encode_depth`.
        encoding: encoding used when encoding.
        depth_max: saturation depth used when encoding.

    Returns:
        Float depression depth in meters (saturated values decode to
        ``depth_max``).
    """
    u = np.asarray(band, dtype=np.float64) / 255.0
    if encoding == "sqrt":
        d = u**2 * depth_max
    else:
        d = np.expm1(u * math.log1p(depth_max)) if math.log1p(depth_max) > 0 else u * 0
    return d


def describe(encoding: str, depth_max: float) -> dict:
    """Return metadata describing the encoding for tile consumers.

    Args:
        encoding: encoding name.
        depth_max: saturation depth in meters.

    Returns:
        Metadata dict for the PMTiles archive.
    """
    return {
        "band": "depression depth",
        "unit": "m (saturated)",
        "dtype": "uint8",
        "encoding": encoding,
        "depth_max_m": depth_max,
        "decode": {
            "sqrt": "depth = (u8 / 255) ** 2 * depth_max_m",
            "log": "depth = expm1((u8 / 255) * log1p(depth_max_m))",
        }[encoding],
    }
