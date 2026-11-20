"""Raster and terrain processing steps: tiles, mosaic, depression fill, clip."""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box, mapping
from whitebox_workflows import WbEnvironment

from storm_water_management.config import DEFAULT_INFILTRATION_FRACTION

WANG_LIU_FLAT_INCREMENT_M = 0.001


def projected_center(longitude: float, latitude: float) -> tuple[float, float]:
    """Convert WGS84 longitude/latitude inputs to SWEREF99 TM coordinates."""
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be finite and between -180 and 180 degrees.")
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be finite and between -90 and 90 degrees.")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    return transformer.transform(longitude, latitude)


def select_tiles(
    input_dir: Path, analysis_bounds: tuple[float, float, float, float]
) -> list[Path]:
    """Find tiles intersecting the analysis bounds and verify coverage/alignment."""
    tile_paths = sorted(input_dir.rglob("*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No .tif elevation tiles found under {input_dir}.")

    left, bottom, right, top = analysis_bounds
    selected: list[Path] = []
    reference_crs = None
    reference_resolution = None
    reference_nodata = None
    for path in tile_paths:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"Elevation tile has no CRS: {path}")
            if reference_crs is None:
                reference_crs = dataset.crs
                reference_resolution = dataset.res
                reference_nodata = dataset.nodata
            elif dataset.crs != reference_crs:
                raise ValueError(f"Elevation tile has a different CRS: {path}")
            elif dataset.res != reference_resolution:
                raise ValueError(f"Elevation tile has a different resolution: {path}")
            elif dataset.nodata != reference_nodata:
                raise ValueError(f"Elevation tile has a different nodata value: {path}")

            bounds = dataset.bounds
            if (
                bounds.right >= left
                and bounds.left <= right
                and bounds.top >= bottom
                and bounds.bottom <= top
            ):
                selected.append(path)

    if not selected:
        raise ValueError(
            f"No elevation tiles intersect the analysis extent {analysis_bounds}."
        )
    selected_bounds = []
    for path in selected:
        with rasterio.open(path) as dataset:
            selected_bounds.append(dataset.bounds)
    coverage = (
        min(bounds.left for bounds in selected_bounds),
        min(bounds.bottom for bounds in selected_bounds),
        max(bounds.right for bounds in selected_bounds),
        max(bounds.top for bounds in selected_bounds),
    )
    if (
        coverage[0] > left
        or coverage[1] > bottom
        or coverage[2] < right
        or coverage[3] < top
    ):
        raise ValueError(
            "Downloaded tiles do not cover the full buffered analysis extent. "
            f"Required {analysis_bounds}; available tile footprint is {coverage}."
        )
    return selected


def create_buffered_mosaic(
    tile_paths: list[Path],
    analysis_bounds: tuple[float, float, float, float],
    output_path: Path,
) -> Path:
    """Mosaic selected tiles, limited to the buffered calculation bounds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        datasets = [stack.enter_context(rasterio.open(path)) for path in tile_paths]
        mosaic, transform = merge(
            datasets, bounds=analysis_bounds, nodata=datasets[0].nodata
        )
        metadata = datasets[0].meta.copy()
    metadata.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "lzw",
            "tiled": True,
            "bigtiff": "IF_SAFER",
        }
    )
    with rasterio.open(output_path, "w", **metadata) as destination:
        destination.write(mosaic)
    return output_path


def run_depression_analysis(
    dem_path: Path,
    output_path: Path,
    contributing_area_path: Path | None = None,
    upslope_storage_path: Path | None = None,
    d8_pointer_path: Path | None = None,
    d8_contributing_area_path: Path | None = None,
) -> Path:
    """Calculate depression depth and optional rainfall-scenario rasters."""
    requested_outputs = (
        contributing_area_path,
        upslope_storage_path,
        d8_pointer_path,
        d8_contributing_area_path,
    )
    if any(path is None for path in requested_outputs) and any(
        path is not None for path in requested_outputs
    ):
        raise ValueError(
            "All flow-accumulation and depression-storage outputs must be requested together."
        )
    environment = WbEnvironment()
    environment.verbose = False
    environment.working_directory = str(dem_path.parent)
    dem = environment.read_raster(dem_path.name)
    # Exact depression depth is used for ponding and runoff calculations, but not for flow accumulation.
    exact_fill = environment.fill_depressions_wang_and_liu(dem, fix_flats=False)
    depth = environment.raster_calculator("('dem_no_deps'-'dem')", [exact_fill, dem])
    environment.write_raster(depth, output_path.name)
    if (
        contributing_area_path is not None
        and upslope_storage_path is not None
        and d8_pointer_path is not None
        and d8_contributing_area_path is not None
    ):
        # Filling with flat increment only for D8-flow.
        filled_dem = environment.fill_depressions_wang_and_liu(
            dem, flat_increment=WANG_LIU_FLAT_INCREMENT_M
        )
        contributing_area = environment.fd8_flow_accum(
            filled_dem, out_type="catchment area", log_transform=False, clip=False
        )
        upslope_storage = environment.upslope_depression_storage(dem)
        d8_pointer = environment.d8_pointer(filled_dem)
        d8_contributing_area = environment.d8_flow_accum(
            filled_dem, out_type="catchment area", log_transform=False, clip=False
        )
        environment.write_raster(contributing_area, contributing_area_path.name)
        environment.write_raster(upslope_storage, upslope_storage_path.name)
        environment.write_raster(d8_pointer, d8_pointer_path.name)
        environment.write_raster(d8_contributing_area, d8_contributing_area_path.name)
    return output_path


def effective_rainfall_m(
    rainfall_mm: float,
    infiltration_fraction: float | None = None,
    infiltration_capacity_mm_h: float | None = None,
    rainfall_duration_hours: float | None = None,
) -> float:
    """Return event rainfall remaining after the uniform infiltration loss.

    Without an infiltration capacity, an unspecified ``infiltration_fraction``
    falls back to the shared ``DEFAULT_INFILTRATION_FRACTION``.
    """
    if not np.isfinite(rainfall_mm) or rainfall_mm < 0:
        raise ValueError("Rainfall depth must be a finite, non-negative value.")
    if infiltration_capacity_mm_h is not None:
        if (
            not np.isfinite(infiltration_capacity_mm_h)
            or infiltration_capacity_mm_h < 0
        ):
            raise ValueError("Infiltration capacity must be finite and non-negative.")
        if (
            rainfall_duration_hours is None
            or not np.isfinite(rainfall_duration_hours)
            or rainfall_duration_hours <= 0
        ):
            raise ValueError(
                "A positive rainfall duration is required with an infiltration capacity."
            )
        infiltrated_mm = infiltration_capacity_mm_h * rainfall_duration_hours
    else:
        fraction = (
            DEFAULT_INFILTRATION_FRACTION
            if infiltration_fraction is None
            else infiltration_fraction
        )
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("The infiltration fraction must be between 0 and 1.")
        infiltrated_mm = rainfall_mm * fraction
    return max(0.0, rainfall_mm - infiltrated_mm) / 1000.0


def clip_to_area(
    raster_path: Path,
    output_path: Path,
    area_bounds: tuple[float, float, float, float],
) -> Path:
    """Mask and crop an analysis raster to the exact rectangular area of interest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as source:
        nodata = source.nodata if source.nodata is not None else 0
        clipped, transform = mask(
            source, [mapping(box(*area_bounds))], crop=True, filled=True, nodata=nodata
        )
        metadata = source.meta.copy()
    metadata.update(
        height=clipped.shape[1],
        width=clipped.shape[2],
        transform=transform,
        nodata=nodata,
        compress="lzw",
        tiled=True,
    )
    with rasterio.open(output_path, "w", **metadata) as destination:
        destination.write(clipped)
    return output_path
