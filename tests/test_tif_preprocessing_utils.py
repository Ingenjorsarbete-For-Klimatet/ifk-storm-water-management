"""Tests for pre processing."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from storm_water_management.tif_preprocessing_utils import concat_tif_in_folder


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


def test_concat_creates_file(temp_tif_folder):
    """Test att concat_tif_in_folder skapar en TIFF-fil."""
    concat_tif_in_folder(str(temp_tif_folder), plot_merge=False)
    output_file = temp_tif_folder / "tifs.tif"
    assert output_file.exists(), "Merged TIFF file should have been created."

    # Check size
    with rasterio.open(output_file) as src:
        merged_data = src.read(1)

    individual_file = temp_tif_folder / "file_0.tif"
    with rasterio.open(individual_file) as src:
        single_data = src.read(1)

    assert merged_data.shape[0] >= single_data.shape[0]
    assert merged_data.shape[1] >= single_data.shape[1]


def test_no_tif_files(tmp_path):
    """Test empty folder."""
    empty_folder = tmp_path / "empty"
    empty_folder.mkdir()

    with pytest.raises(ValueError, match="No .tif-files in:"):
        concat_tif_in_folder(str(empty_folder))
