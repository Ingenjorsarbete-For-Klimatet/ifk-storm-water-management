"""Depression filling via the priority-flood algorithm.

This is a numpy/numba implementation of the same class of algorithm as
Whitebox's ``fill_depressions_wang_and_liu`` (Wang & Liu 2006, Barnes
et al. 2014): a priority queue seeded with all boundary cells grows a
raised surface inward, raising any cell lower than its processed
neighbour. The difference between the filled surface and the original
DEM is the depression depth.

NaN cells act as drains: they are seeded with ``-inf`` so surrounding
data drains into them. This keeps un-surveyed holes ( lakes are data,
holes are missing postings) from creating artificial depressions at
hole edges.
"""

import numba
import numpy as np


@numba.njit(cache=True)
def _heap_push(keys, values, size, key, value):
    """Push a (key, value) pair onto a binary min-heap."""
    i = size
    keys[i] = key
    values[i] = value
    while i > 0:
        parent = (i - 1) // 2
        if keys[parent] <= keys[i]:
            break
        keys[parent], keys[i] = keys[i], keys[parent]
        values[parent], values[i] = values[i], values[parent]
        i = parent
    return size + 1


@numba.njit(cache=True)
def _heap_pop(keys, values, size):
    """Pop the smallest (key, value) pair from a binary min-heap."""
    top_key = keys[0]
    top_value = values[0]
    size -= 1
    keys[0] = keys[size]
    values[0] = values[size]
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and keys[left] < keys[smallest]:
            smallest = left
        if right < size and keys[right] < keys[smallest]:
            smallest = right
        if smallest == i:
            break
        keys[i], keys[smallest] = keys[smallest], keys[i]
        values[i], values[smallest] = values[smallest], values[i]
        i = smallest
    return top_key, top_value, size


@numba.njit(cache=True)
def _fill(dem, filled):
    """Grow the filled surface over ``dem`` into ``filled``."""
    h, w = dem.shape
    n = h * w

    heap_keys = np.empty(n, dtype=np.float32)
    heap_values = np.empty(n, dtype=np.int32)
    closed = np.zeros(n, dtype=np.uint8)

    heap_size = 0

    # Seed the heap with NaN cells (drains, pop lowest) and border cells.
    for r in range(h):
        for c in range(w):
            cell = r * w + c
            z = dem[r, c]
            border = r == 0 or c == 0 or r == h - 1 or c == w - 1
            if np.isnan(z) or border:
                closed[cell] = True
                key = np.float32(-np.inf) if np.isnan(z) else np.float32(z)
                heap_size = _heap_push(heap_keys, heap_values, heap_size, key, cell)

    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    while heap_size > 0:
        _, cell, heap_size = _heap_pop(heap_keys, heap_values, heap_size)
        row = cell // w
        col = cell - row * w
        z_here = np.float32(filled[row, col])
        # NaN cells act as sinks at -inf: never raise their neighbours.
        if np.isnan(z_here):
            z_here = np.float32(-np.inf)
        for dr in range(8):
            nr = row + offsets[dr][0]
            nc = col + offsets[dr][1]
            if nr < 0 or nc < 0 or nr >= h or nc >= w:
                continue
            ncell = nr * w + nc
            if closed[ncell]:
                continue
            closed[ncell] = True
            z_n = dem[nr, nc]
            if np.isnan(z_n):
                filled[nr, nc] = np.float32(np.nan)
                heap_size = _heap_push(
                    heap_keys, heap_values, heap_size, np.float32(-np.inf), ncell
                )
            else:
                z_new = z_n if z_n >= z_here else z_here
                filled[nr, nc] = z_new
                heap_size = _heap_push(
                    heap_keys, heap_values, heap_size, np.float32(z_new), ncell
                )


def depression_depth(elevation: np.ndarray) -> np.ndarray:
    """Compute depression depth of a DEM.

    Args:
        elevation: north-up float32 array, NaN where no data.

    Returns:
        Depression depth (filled minus original) in meters, 0 outside
        depressions and NaN where the input is NaN.
    """
    dem = np.ascontiguousarray(elevation, dtype=np.float32)
    filled = dem.copy()
    _fill(dem, filled)
    depth = filled - dem
    depth[dem != dem] = np.nan
    depth[depth < 0] = 0.0
    return depth
