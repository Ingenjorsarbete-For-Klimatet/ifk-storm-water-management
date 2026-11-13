"""Delta depression-depth tiles into a raster PMTiles archive.

The block-wise SWEREF99 TM depth rasters are reprojected (nearest
neighbour) into Web Mercator at a single zoom level, cut into 256 px
tiles, encoded as lossless grayscale WebP or AVIF and written into one
PMTiles archive. Tiles that carry no elevation data at all are
omitted; PMTiles run-length/hashing de-duplicates the many all-zero
tiles.

Consume in maplibregl with the pmtiles protocol adapter::

    "sources": {
        "depressions": {
            "type": "raster",
            "url": "pmtiles://https://.../depressions_z11.pmtiles"
        }
    }
"""

import io
import math

import numpy as np
import rasterio.warp
from affine import Affine
from PIL import Image
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import write

from storm_water_management import depth_encode

WEB_MERCATOR_WORLD = 20037508.342789244
TILE_SIZE = 256


def meters_per_pixel(zoom: int) -> float:
    """Web Mercator ground resolution at the equator for a zoom level.

    Args:
        zoom: zoom level.

    Returns:
        Meters per pixel at the equator.
    """
    return 2 * WEB_MERCATOR_WORLD / (TILE_SIZE * 2**zoom)


def tile_to_lonlat(z: int, x: int, y: int, corner: str) -> tuple[float, float]:
    """Lon/lat of a tile corner.

    Args:
        z: zoom level.
        x: tile x (north-origin scheme).
        y: tile y (north-origin scheme).
        corner: ``"nw"`` or ``"se"``.

    Returns:
        ``(lon, lat)`` in degrees.
    """
    n = 2**z
    if corner == "nw":
        xf, yf = x, y
    else:
        xf, yf = x + 1, y + 1
    lon = xf / n * 360.0 - 180.0
    yr = math.pi * (1 - 2 * yf / n)
    lat = math.degrees(2 * math.atan(math.exp(yr)) - math.pi / 2)
    return lon, lat


def _tile_ranges(bounds_3857, zoom: int) -> tuple[int, int, int, int]:
    """Tile x/y range covering a Web Mercator bounding box.

    Args:
        bounds_3857: ``(west, south, east, north)`` in EPSG:3857.
        zoom: zoom level.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` inclusive tile coordinates,
        north-origin (XYZ/Google scheme).
    """
    n = 2**zoom
    west, south, east, north = bounds_3857
    x_min = int(np.floor((west + WEB_MERCATOR_WORLD) / (2 * WEB_MERCATOR_WORLD) * n))
    x_max = int(np.ceil((east + WEB_MERCATOR_WORLD) / (2 * WEB_MERCATOR_WORLD) * n)) - 1
    y_min = int(np.floor((WEB_MERCATOR_WORLD - north) / (2 * WEB_MERCATOR_WORLD) * n))
    y_max = (
        int(np.ceil((WEB_MERCATOR_WORLD - south) / (2 * WEB_MERCATOR_WORLD) * n)) - 1
    )
    return x_min, y_min, x_max, y_max


def _alloc_window(bounds_3857, zoom: int):
    """Destination raster window covering the tiles around a bbox.

    Args:
        bounds_3857: bbox in EPSG:3857.
        zoom: zoom level.

    Returns:
        ``(transform, height, width, x0, y0)`` of a raster aligned to
        the tile grid starting tile ``(x0, y0)``.
    """
    x_min, y_min, x_max, y_max = _tile_ranges(bounds_3857, zoom)
    res = meters_per_pixel(zoom)
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE
    transform = Affine(
        res,
        0,
        -WEB_MERCATOR_WORLD + x_min * TILE_SIZE * res,
        0,
        -res,
        WEB_MERCATOR_WORLD - y_min * TILE_SIZE * res,
    )
    return transform, height, width, x_min, y_min


def block_to_tiles(
    depth: np.ndarray,
    x_min_cell: int,
    y_min_cell: int,
    zoom: int,
    codec: str,
    encoding: str,
    depth_max: float,
    resolution: float = 50.0,
) -> dict[tuple[int, int, int], bytes]:
    """Reproject one SWEREF99 TM depth block into encoded uint8 mercator tiles.

    Args:
        depth: north-up float32 depression depth array, NaN where no
            data.
        x_min_cell: west cell index (SWEREF99 TM 50 m grid).
        y_min_cell: south cell index (SWEREF99 TM 50 m grid).
        zoom: single output zoom level.
        codec: lossless tile codec, ``"webp"`` or ``"avif"``.
        encoding: depth encoding (``sqrt`` or ``log``).
        depth_max: saturation depth in meters.
        resolution: source resolution in meters.

    Returns:
        Dict mapping ``(zoom, x, y)`` (north-origin scheme) to encoded
        lossless tile bytes. Only tiles containing at least one data
        cell are returned.
    """
    src_height, src_width = depth.shape
    src_transform = Affine(
        resolution,
        0,
        x_min_cell * resolution,
        0,
        -resolution,
        (y_min_cell + src_height) * resolution,
    )
    w, s, e, n = rasterio.warp.array_bounds(src_height, src_width, src_transform)
    bounds_3857 = rasterio.warp.transform_bounds("EPSG:3006", "EPSG:3857", w, s, e, n)
    dst_transform, height, width, tx0, ty0 = _alloc_window(bounds_3857, zoom)

    dst = np.full((height, width), np.nan, dtype=np.float32)
    rasterio.warp.reproject(
        source=depth,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:3006",
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=rasterio.warp.Resampling.nearest,
    )

    tiles = {}
    for ty in range(height // TILE_SIZE):
        for tx in range(width // TILE_SIZE):
            sl = (
                slice(ty * TILE_SIZE, (ty + 1) * TILE_SIZE),
                slice(tx * TILE_SIZE, (tx + 1) * TILE_SIZE),
            )
            tile = dst[sl]
            if np.all(np.isnan(tile)):
                continue
            band = depth_encode.encode_depth(tile, encoding, depth_max)
            tiles[(zoom, tx + tx0, ty + ty0)] = encode_tile(band, codec)
    return tiles


def encode_tile(tile: np.ndarray, codec: str) -> bytes:
    """Encode a uint8 tile image losslessly.

    Both codecs are verified pixel-exact for uint8 grayscale:

    - ``webp``: lossless WebP (method 4). Fast and compact; supported
      by all browsers and maplibre raster sources.
    - ``avif``: lossless AVIF needs ``quality=100`` and
      ``range="full"`` to be exact, and is both slower and usually
      larger than WebP on this content.

    Args:
        tile: uint8 tile array.
        codec: ``"webp"`` or ``"avif"``.

    Returns:
        Encoded tile bytes.
    """
    img = Image.fromarray(tile, mode="L")
    buf = io.BytesIO()
    if codec == "webp":
        img.save(buf, format="WEBP", lossless=True, quality=100, method=4)
    elif codec == "avif":
        import pillow_avif  # noqa: F401  (registers the AVIF codec)

        img.save(buf, format="AVIF", lossless=True, quality=100, range="full")
    else:
        raise ValueError(f"Unknown codec {codec!r}")
    return buf.getvalue()


def write_pmtiles(
    filepath: str,
    tiles: dict[tuple[int, int, int], bytes],
    *,
    codec: str,
    zoom: int,
    encoding: str,
    depth_max: float,
    resolution: float = 50.0,
    metadata_extra: dict | None = None,
) -> dict:
    """Write tile bytes into a PMTiles v3 archive.

    Args:
        filepath: output ``.pmtiles`` path.
        tiles: dict mapping ``(zoom, x, y)`` to encoded tile bytes.
        codec: tile codec, ``"webp"`` or ``"avif"``.
        zoom: zoom level of the archive (single level).
        encoding: depth encoding used (see :mod:`depth_encode`).
        depth_max: saturation depth in meters.
        resolution: source grid resolution in meters.
        metadata_extra: additional metadata entries.

    Returns:
        Summary dict with tile counts and byte totals.
    """
    tile_type = {"webp": TileType.WEBP, "avif": TileType.AVIF}[codec]

    metadata = {
        "name": f"Storm water depression depth (Lantmäteriet {resolution:g} m grid)",
        "description": "Depression (sink) depth derived from Lantmäteriet "
        f"Höjddata grid {resolution:g} m elevation model.",
        "attribution": "<a href='https://www.lantmateriet.se'>Lantmäteriet</a>",
        "encoding": depth_encode.describe(encoding, depth_max),
    }
    if metadata_extra:
        metadata.update(metadata_extra)

    sorted_tiles = sorted(tiles.items(), key=lambda kv: zxy_to_tileid(*kv[0]))

    lon_lats = [tile_to_lonlat(z, x, y, c) for (z, x, y) in tiles for c in ("nw", "se")]
    lons = [p[0] for p in lon_lats]
    lats = [p[1] for p in lon_lats]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    header = {
        "version": 3,
        "root_offset": 0,
        "root_length": 0,
        "metadata_offset": 0,
        "metadata_length": 0,
        "tile_data_offset": 0,
        "tile_data_length": 0,
        "clustered": True,
        "internal_compression": Compression.GZIP,
        "tile_compression": Compression.NONE,
        "tile_type": tile_type,
        "min_zoom": zoom,
        "max_zoom": zoom,
        "center_zoom": zoom,
        "min_lon_e7": int(lon_min * 1e7),
        "min_lat_e7": int(lat_min * 1e7),
        "max_lon_e7": int(lon_max * 1e7),
        "max_lat_e7": int(lat_max * 1e7),
    }

    with write(filepath) as f:
        for (z, x, y), data in sorted_tiles:
            f.write_tile(zxy_to_tileid(z, x, y), data)
        f.finalize(header, metadata)

    return {
        "tiles": len(tiles),
        "unique_tile_bytes": sum(len(v) for v in tiles.values()),
    }
