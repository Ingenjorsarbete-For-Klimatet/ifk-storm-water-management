"""Tests for pre processing."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def temp_tif_folder(tmp_path):
    """Create temp folder with tif-files."""
    folder = tmp_path / "tifs"
    folder.mkdir()

    for i in range(2):
        data = np.array([[i + 1, i + 2], [i + 3, i + 4]], dtype=np.float32)
        transform = from_origin(0, 2, 1, 1)
        tif_path = folder / f"file_{i}.tif"

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    return folder
