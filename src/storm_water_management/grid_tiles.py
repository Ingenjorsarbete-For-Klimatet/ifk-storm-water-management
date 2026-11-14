"""Reading of Lantmäteriet 1 m GeoTIFF grid tiles as SWEREF 99 TM rasters.

The 1 m elevation grid ("Höjddata grid 1 m") is delivered as a folder
of axis-aligned GeoTIFF tiles (typically 2.5 x 2.5 km). Unlike the
50 m XYZ deliveries the lattice is not rotated against SWEREF 99 TM,
so no binning of postings is needed: cell indices on the global 1 m
SWEREF 99 TM lattice are simply the rounded easting/northing, and a
tile's raster is taken over as it is read.

Blocks (tiles) are kept in an LRU cache and assembled with a halo of
cells from neighbouring tiles, mirroring
:mod:`storm_water_management.sweden_grid`, so that depressions
spanning tile borders drain consistently.

Note:
    The halo is specified in cells (1 cell = 1 m). The default of
    1000 cells (1 km) keeps peak memory per 2.5 km tile comfortably
    below 0.5 GB; widen it for larger cross-border depressions at
    the cost of RAM.
"""

import os
from collections import OrderedDict

import numpy as np
import rasterio

RESOLUTION_M = 1
NODATA = np.float32("nan")

# Default halo: distance (in cells) kept from neighbouring tiles, so
# depressions spanning tile borders drain consistently. 1000 cells = 1 km.
HALO_CELLS = 1000

# Spatial index bucket size in cells; tiles are registered in every
# bucket they touch. 4096 cells keeps a 2.5 km tile in at most four
# buckets while making neighbour queries a dictionary lookup.
BUCKET_CELLS = 4096


def tile_names(data_folder: str) -> list[str]:
    """List all tile names (GeoTIFF stems) available in a data folder.

    Args:
        data_folder: folder containing ``<name>.tif`` tile files.

    Returns:
        Sorted list of tile names.
    """
    return sorted(f[:-4] for f in os.listdir(data_folder) if f.lower().endswith(".tif"))


def _tile_path(data_folder: str, tile: str) -> str:
    return os.path.join(data_folder, f"{tile}.tif")


class GridTileStore:
    """LRU cache of parsed elevation tiles with a spatial index.

    Each GeoTIFF tile is read once and kept as a float32 raster on the
    global 1 m SWEREF 99 TM lattice (nodata as NaN). Tile bounds are
    indexed in cell-coordinate buckets so the neighbours of any window
    can be found without opening every raster.
    """

    def __init__(self, data_folder: str, max_tiles: int = 12) -> None:
        """Create a grid tile store.

        Args:
            data_folder: folder with the GeoTIFF tiles.
            max_tiles: number of parsed tiles kept in memory.
        """
        self.data_folder = data_folder
        self.max_tiles = max_tiles
        self._cache: OrderedDict[str, tuple[np.ndarray, int, int]] = OrderedDict()
        self.available = set(tile_names(data_folder))
        # tile name -> cell index bounds (x_min, y_min, x_max, y_max)
        self._extents: dict[str, tuple[int, int, int, int]] = {
            tile: self._read_extents(tile) for tile in self.available
        }
        # bucket cell coordinates -> names of tiles touching the bucket
        self._buckets: dict[tuple[int, int], set[str]] = {}
        for tile, (x_min, y_min, x_max, y_max) in self._extents.items():
            for bx in range(x_min // BUCKET_CELLS, (x_max - 1) // BUCKET_CELLS + 1):
                for by in range(y_min // BUCKET_CELLS, (y_max - 1) // BUCKET_CELLS + 1):
                    self._buckets.setdefault((bx, by), set()).add(tile)

    def _read_extents(self, tile: str) -> tuple[int, int, int, int]:
        """Read one tile's cell index bounds from its GeoTIFF header."""
        with rasterio.open(_tile_path(self.data_folder, tile)) as dataset:
            transform = dataset.transform
            if transform.b != 0 or transform.d != 0:
                raise ValueError(f"Tile {tile} is not axis-aligned")
            if not np.isclose(transform.a, RESOLUTION_M) or not np.isclose(
                -transform.e, RESOLUTION_M
            ):
                raise ValueError(f"Tile {tile} is not a {RESOLUTION_M} m grid tile")
            x_min = round(transform.c / RESOLUTION_M)
            y_max = round(transform.f / RESOLUTION_M)
            width, height = dataset.width, dataset.height
        return x_min, y_max - height, x_min + width, y_max

    def full(self, tile: str) -> tuple[np.ndarray, int, int]:
        """Return the full raster of a tile (cached).

        Args:
            tile: tile name, the GeoTIFF file stem.

        Returns:
            ``(elevation, x_min, y_min)`` with NaN outside data;
            elevation is north-up, ``(x_min, y_min)`` are south-west
            corner cell indices of the returned array.
        """
        if tile not in self._cache:
            self._cache[tile] = self._read_tile_grid(tile)
            if len(self._cache) > self.max_tiles:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(tile)
        return self._cache[tile]

    def _read_tile_grid(self, tile: str) -> tuple[np.ndarray, int, int]:
        """Read one tile as a north-up raster on the global lattice."""
        with rasterio.open(_tile_path(self.data_folder, tile)) as dataset:
            band = dataset.read(1, masked=True)
        grid = np.ma.filled(band, np.nan).astype(np.float32)
        x_min, y_min, _, _ = self._extents[tile]
        return grid, x_min, y_min

    def neighbours(self, window: tuple[int, int, int, int]) -> list[str]:
        """List cached tiles whose extents touch a cell index window.

        Args:
            window: ``(x_min, y_min, x_max, y_max)`` cell indices.

        Returns:
            Names of tiles intersecting the window, arbitrary order.
        """
        wx_min, wy_min, wx_max, wy_max = window
        names: set[str] = set()
        for bx in range(wx_min // BUCKET_CELLS, (wx_max - 1) // BUCKET_CELLS + 1):
            for by in range(wy_min // BUCKET_CELLS, (wy_max - 1) // BUCKET_CELLS + 1):
                names.update(self._buckets.get((bx, by), ()))
        return [
            n for n in names if n in self._extents and self._tile_intersects(n, window)
        ]

    def _tile_intersects(self, tile: str, window: tuple[int, int, int, int]) -> bool:
        """Report whether a tile's extent touches a cell index window."""
        tx_min, ty_min, tx_max, ty_max = self._extents[tile]
        wx_min, wy_min, wx_max, wy_max = window
        return (
            tx_min < wx_max and tx_max > wx_min and ty_min < wy_max and ty_max > wy_min
        )

    def window(
        self, tile: str, window: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, int, int]:
        """Return the part of a tile inside a cell index window.

        Args:
            tile: tile name.
            window: ``(x_min, y_min, x_max, y_max)`` cell indices.

        Returns:
            ``(elevation, x_min, y_min)`` of the intersection, possibly
            an empty array.
        """
        arr, bx_min, by_min = self.full(tile)
        wx_min, wy_min, wx_max, wy_max = window
        h, w = arr.shape
        bx_max, by_max = bx_min + w, by_min + h
        x0 = max(bx_min, wx_min)
        x1 = min(bx_max, wx_max)
        y0 = max(by_min, wy_min)
        y1 = min(by_max, wy_max)
        if x0 >= x1 or y0 >= y1:
            return np.empty((0, 0), dtype=np.float32), x0, y0
        # rows are north-up: row 0 is the highest cell index
        return arr[by_max - y1 : by_max - y0, x0 - bx_min : x1 - bx_min], x0, y0


def assemble_block(
    tile: str,
    store: GridTileStore,
    halo_cells: int = HALO_CELLS,
) -> tuple[np.ndarray, int, int]:
    """Assemble one tile with halo from neighbouring tiles.

    The tile's own footprint is padded with ``halo_cells`` cells of
    elevation taken from surrounding tiles (as far as they exist).
    This makes depressions that straddle tile borders drain
    consistently.

    Args:
        tile: tile name, the GeoTIFF file stem.
        store: tile store used for reading tiles.
        halo_cells: number of 1 m cells of padding.

    Returns:
        Tuple of ``(elevation, x_min, y_min)`` as a north-up float32
        array with NaN outside known data and south-west corner cell
        indices of the returned array.
    """
    own, x_min, y_min = store.full(tile)
    if own.size == 0:
        raise ValueError(f"Tile {tile} has no data")
    height, width = own.shape

    x_min_i = x_min - halo_cells
    y_min_i = y_min - halo_cells
    window = (
        x_min_i,
        y_min_i,
        x_min + width + halo_cells,
        y_min + height + halo_cells,
    )

    pad_h = height + 2 * halo_cells
    pad_w = width + 2 * halo_cells
    combined = np.full((pad_h, pad_w), NODATA, dtype=np.float32)
    combined[halo_cells : halo_cells + height, halo_cells : halo_cells + width] = own

    for neighbour in store.neighbours(window):
        if neighbour == tile:
            continue
        arr, nx_min, ny_min = store.window(neighbour, window)
        if arr.size == 0:
            continue
        tx = nx_min - x_min_i
        ty = pad_h - (ny_min + arr.shape[0] - y_min_i)
        target = combined[ty : ty + arr.shape[0], tx : tx + arr.shape[1]]
        empty = np.isnan(target)
        target[empty] = arr[empty]

    return combined, x_min_i, y_min_i


def open_store(data_folder: str) -> GridTileStore:
    """Open a grid tile store for a data folder.

    Args:
        data_folder: folder with the GeoTIFF tiles.

    Returns:
        The tile store.
    """
    return GridTileStore(data_folder)
