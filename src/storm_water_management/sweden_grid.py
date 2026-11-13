"""Reading of Lantmäteriet 50 m grid XYZ deliveries as SWEREF 99 TM rasters.

The deliveries come as one ``<north-band>_<east-band>.zip`` file per
(roughly) 100 km block of the SWEREF 99 TM grid. Each zip contains a
single ``x y z`` text file with comma decimal separators, one point
per line.

The postings form a 50 m lattice that is rotated slightly (sub-degree)
relative to the SWEREF 99 TM axes, so the points cannot be read as a
pre-gridded raster. Instead every point is binned onto the global
SWEREF 99 TM 50 m lattice (cell indices ``round(x / 50)``,
``round(y / 50)``), which covers > 99 % of the lattice cells.
"""

import os
import zipfile
from collections import OrderedDict

import numpy as np
import pandas as pd

RESOLUTION_M = 50
NODATA = np.float32("nan")

# Distance (in cells) kept from neighbouring blocks when assembling a
# block with halo, so that depressions spanning block borders are
# treated consistently. 200 cells = 10 km.
HALO_CELLS = 200


def block_names(data_folder: str) -> list[str]:
    """List all block names (zip stems) available in a data folder.

    Args:
        data_folder: folder containing ``<name>.zip`` block files.

    Returns:
        Sorted list of block names.
    """
    return sorted(
        f[:-4]
        for f in os.listdir(data_folder)
        if f.lower().endswith(".zip") and f.count("_") == 1
    )


def _zip_path(data_folder: str, block: str) -> str:
    return os.path.join(data_folder, f"{block}.zip")


def _read_xyz(zip_file: str) -> pd.DataFrame:
    """Read one block's postings from its zip as a DataFrame."""
    with zipfile.ZipFile(zip_file) as zf:
        member = [n for n in zf.namelist() if n.lower().endswith(".xyz")][0]
        with zf.open(member) as f:
            df = pd.read_csv(
                f,
                sep=r"\s+",
                header=None,
                names=["x", "y", "z"],
                decimal=",",
                dtype=np.float64,
            )
    return df


def _read_block_grid(block: str, data_folder: str) -> tuple[np.ndarray, int, int]:
    """Read one block as a north-up raster on the global 50 m lattice.

    Returns:
        Tuple of ``(elevation, x_min, y_min)`` where elevation is a
        north-up float32 array (nodata as NaN) and ``(x_min, y_min)``
        are south-west corner cell indices of the returned array.
    """
    df = _read_xyz(_zip_path(data_folder, block))
    ix = np.rint(df["x"].to_numpy() / RESOLUTION_M).astype(np.int64)
    iy = np.rint(df["y"].to_numpy() / RESOLUTION_M).astype(np.int64)
    z = df["z"].to_numpy()

    x_min, y_min = int(ix.min()), int(iy.min())
    x_max, y_max = int(ix.max()) + 1, int(iy.max()) + 1
    width, height = x_max - x_min, y_max - y_min

    grid = np.full((height, width), NODATA, dtype=np.float32)
    grid[height - 1 - (iy - y_min), ix - x_min] = z
    return grid, x_min, y_min


class BlockStore:
    """LRU cache of parsed elevation blocks.

    Each block is parsed from its zip once and kept as a binned
    raster so neighbours can be sliced cheaply when assembling
    halo-padded blocks.
    """

    def __init__(self, data_folder: str, max_blocks: int = 24) -> None:
        """Create a block store.

        Args:
            data_folder: folder containing the block zips.
            max_blocks: number of parsed blocks kept in memory.
        """
        self.data_folder = data_folder
        self.max_blocks = max_blocks
        self._cache: OrderedDict[str, tuple[np.ndarray, int, int]] = OrderedDict()
        self.available = set(block_names(data_folder))

    def full(self, block: str) -> tuple[np.ndarray, int, int]:
        """Return the full binned raster of a block (cached).

        Args:
            block: block name, e.g. ``"63_6"``.

        Returns:
            ``(elevation, x_min, y_min)`` with NaN outside data.
        """
        if block not in self._cache:
            self._cache[block] = _read_block_grid(block, self.data_folder)
            if len(self._cache) > self.max_blocks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(block)
        return self._cache[block]

    def window(
        self, block: str, window: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, int, int]:
        """Return the part of a block inside a cell index window.

        Args:
            block: block name.
            window: ``(x_min, y_min, x_max, y_max)`` cell indices.

        Returns:
            ``(elevation, x_min, y_min)`` of the intersection, possibly
            an empty array.
        """
        arr, bx_min, by_min = self.full(block)
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
    block: str,
    store: BlockStore,
    halo_cells: int = HALO_CELLS,
) -> tuple[np.ndarray, int, int]:
    """Assemble one block with halo from neighbouring blocks.

    The block's own footprint is padded with ``halo_cells`` cells of
    elevation taken from surrounding blocks (as far as they exist).
    This makes depressions that straddle block borders drain
    consistently.

    Args:
        block: block name, e.g. ``"63_6"``.
        store: block store used for reading blocks.
        halo_cells: number of 50 m cells of padding.

    Returns:
        Tuple of ``(elevation, x_min, y_min)`` as a north-up float32
        array with NaN outside known data and south-west corner cell
        indices of the returned array.
    """
    own, x_min, y_min = store.full(block)
    if own.size == 0:
        raise ValueError(f"Block {block} has no data")
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

    north_band, east_band = block.split("_")
    for dn in (-1, 0, 1):
        for de in (-1, 0, 1):
            if dn == 0 and de == 0:
                continue
            neighbour = f"{int(north_band) + dn}_{int(east_band) + de}"
            if neighbour not in store.available:
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


def core_slice(array: np.ndarray, halo_cells: int = HALO_CELLS) -> np.ndarray:
    """Cut the block's core out of a halo-padded array.

    Args:
        array: halo-padded array from :func:`assemble_block`.
        halo_cells: padding that was used when assembling.

    Returns:
        View of the block's own footprint.
    """
    return array[halo_cells:-halo_cells, halo_cells:-halo_cells]


def open_store(data_folder: str) -> BlockStore:
    """Open a block store for a data folder.

    Args:
        data_folder: folder with the block zips.

    Returns:
        The block store.
    """
    return BlockStore(data_folder)
