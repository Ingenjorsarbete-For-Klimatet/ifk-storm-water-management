"""Reusable raster and terrain-workflow helpers for Enköping analysis."""

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
    filled_dem = environment.fill_depressions_wang_and_liu(
        dem, flat_increment=WANG_LIU_FLAT_INCREMENT_M
    )
    depth = environment.raster_calculator("('dem_no_deps'-'dem')", [filled_dem, dem])
    environment.write_raster(depth, output_path.name)
    if all(path is not None for path in requested_outputs):
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
    """Return event rainfall remaining after the uniform infiltration loss."""
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
        fraction = 0.5 if infiltration_fraction is None else infiltration_fraction
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("The infiltration fraction must be between 0 and 1.")
        infiltrated_mm = rainfall_mm * fraction
    return max(0.0, rainfall_mm - infiltrated_mm) / 1000.0


def estimate_runoff_volume_m3(
    contributing_area_m2: np.ndarray,
    upslope_storage_depth_m: np.ndarray,
    effective_rain_m: float,
) -> np.ndarray:
    """Legacy per-cell runoff approximation after average upslope storage."""
    runoff_depth_m = np.maximum(effective_rain_m - upslope_storage_depth_m, 0.0)
    return contributing_area_m2 * runoff_depth_m


def estimate_ponding_depth_m(
    upslope_storage_depth_m: np.ndarray,
    local_depression_depth_m: np.ndarray,
    effective_rain_m: float,
) -> np.ndarray:
    """Estimate ponding depth, capped by rainfall and local depression depth."""
    available_water_m = np.maximum(effective_rain_m - upslope_storage_depth_m, 0.0)
    return np.minimum(available_water_m, np.maximum(local_depression_depth_m, 0.0))


def write_estimated_runoff_volume(
    contributing_area_path: Path,
    upslope_storage_path: Path,
    output_path: Path,
    effective_rain_m: float,
) -> Path:
    """Write the legacy per-cell runoff-volume raster in memory-bounded windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(contributing_area_path) as area_source,
        rasterio.open(upslope_storage_path) as storage_source,
    ):
        if (
            area_source.shape != storage_source.shape
            or area_source.transform != storage_source.transform
            or area_source.crs != storage_source.crs
        ):
            raise ValueError(
                "Contributing-area and upslope-storage rasters are not aligned."
            )
        nodata = area_source.nodata if area_source.nodata is not None else -9999.0
        metadata = area_source.profile.copy()
        metadata.pop("blockxsize", None)
        metadata.pop("blockysize", None)
        metadata.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            tiled=True,
        )
        with rasterio.open(output_path, "w", **metadata) as destination:
            for row_off in range(0, area_source.height, 512):
                for col_off in range(0, area_source.width, 512):
                    window = rasterio.windows.Window(
                        col_off,
                        row_off,
                        min(512, area_source.width - col_off),
                        min(512, area_source.height - row_off),
                    )
                    area = area_source.read(1, window=window, masked=True)
                    storage = storage_source.read(1, window=window, masked=True)
                    area_values = np.asarray(area.data, dtype=np.float64)
                    storage_values = np.asarray(storage.data, dtype=np.float64)
                    valid = (
                        ~np.ma.getmaskarray(area)
                        & ~np.ma.getmaskarray(storage)
                        & np.isfinite(area_values)
                        & np.isfinite(storage_values)
                    )
                    result = np.full(area_values.shape, nodata, dtype=np.float32)
                    result[valid] = estimate_runoff_volume_m3(
                        area_values[valid], storage_values[valid], effective_rain_m
                    ).astype(np.float32)
                    destination.write(result, 1, window=window)
    return output_path


def write_estimated_ponding_depth(
    upslope_storage_path: Path,
    depression_depth_path: Path,
    output_path: Path,
    effective_rain_m: float,
) -> Path:
    """Write the legacy local ponding approximation in metre units by window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(upslope_storage_path) as storage_source,
        rasterio.open(depression_depth_path) as depression_source,
    ):
        if (
            storage_source.shape != depression_source.shape
            or storage_source.transform != depression_source.transform
            or storage_source.crs != depression_source.crs
        ):
            raise ValueError(
                "Upslope-storage and depression-depth rasters are not aligned."
            )
        nodata = (
            depression_source.nodata
            if depression_source.nodata is not None
            else -9999.0
        )
        metadata = depression_source.profile.copy()
        metadata.pop("blockxsize", None)
        metadata.pop("blockysize", None)
        metadata.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=nodata,
            compress="lzw",
            tiled=True,
        )
        with rasterio.open(output_path, "w", **metadata) as destination:
            for row_off in range(0, depression_source.height, 512):
                for col_off in range(0, depression_source.width, 512):
                    window = rasterio.windows.Window(
                        col_off,
                        row_off,
                        min(512, depression_source.width - col_off),
                        min(512, depression_source.height - row_off),
                    )
                    storage = storage_source.read(1, window=window, masked=True)
                    depression = depression_source.read(1, window=window, masked=True)
                    storage_values = np.asarray(storage.data, dtype=np.float64)
                    depression_values = np.asarray(depression.data, dtype=np.float64)
                    valid = (
                        ~np.ma.getmaskarray(storage)
                        & ~np.ma.getmaskarray(depression)
                        & np.isfinite(storage_values)
                        & np.isfinite(depression_values)
                    )
                    result = np.full(storage_values.shape, nodata, dtype=np.float32)
                    result[valid] = estimate_ponding_depth_m(
                        storage_values[valid],
                        depression_values[valid],
                        effective_rain_m,
                    ).astype(np.float32)
                    destination.write(result, 1, window=window)
    return output_path


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
