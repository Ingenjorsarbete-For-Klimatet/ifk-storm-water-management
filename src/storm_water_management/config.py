"""Configuration dataclasses for elevation, terrain, and SWMM workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadConfig:
    """Inputs for downloading elevation tiles for a study area."""

    name: str
    center_lon: float | None = None
    center_lat: float | None = None
    radius_km: float = 10.0
    bbox: tuple[float, float, float, float] | None = None
    output_dir: Path | None = None
    collection: str = "dtm-cog"

    def __post_init__(self) -> None:
        """Validate settings independent of the download service."""
        if not self.name.strip():
            raise ValueError("Download location name cannot be empty.")
        if not self.collection.strip():
            raise ValueError("STAC collection cannot be empty.")
        if not math.isfinite(self.radius_km) or self.radius_km <= 0:
            raise ValueError("Download radius must be finite and positive.")
        if self.bbox is None:
            if self.center_lon is None or self.center_lat is None:
                raise ValueError(
                    "Supply either bbox or both center longitude and latitude."
                )
            if not math.isfinite(self.center_lon) or not -180 <= self.center_lon <= 180:
                raise ValueError("Center longitude must be between -180 and 180 degrees.")
            if not math.isfinite(self.center_lat) or not -90 <= self.center_lat <= 90:
                raise ValueError("Center latitude must be between -90 and 90 degrees.")
        else:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            if not -180 <= min_lon < max_lon <= 180:
                raise ValueError("Longitude bounds must satisfy -180 <= min < max <= 180.")
            if not -90 <= min_lat < max_lat <= 90:
                raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            raise TypeError("Download output_dir must be a pathlib.Path or None.")


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs and options for one location-independent terrain analysis."""

    input_dir: Path
    output_dir: Path
    center_lon: float
    center_lat: float
    output_prefix: str = "analysis"
    rainfall_mm: float | None = None
    prepare_basins_only: bool = False
    infiltration_fraction: float | None = None
    infiltration_capacity_mm_h: float | None = None
    rainfall_duration_hours: float | None = None
    minimum_depression_depth_m: float = 0.02
    minimum_basin_area_m2: float = 5.0
    west_km: float = 2.0
    east_km: float = 2.0
    north_km: float = 2.0
    south_km: float = 2.0
    buffer_west_km: float = 1.0
    buffer_east_km: float = 1.0
    buffer_north_km: float = 1.0
    buffer_south_km: float = 1.0
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate settings independent of any specific location."""
        if not self.output_prefix or Path(self.output_prefix).name != self.output_prefix:
            raise ValueError("Output prefix must be a non-empty filename component.")
        if self.rainfall_mm is not None and (
            not math.isfinite(self.rainfall_mm) or self.rainfall_mm <= 0
        ):
            raise ValueError("Rainfall must be finite and greater than zero.")
        if self.prepare_basins_only and self.rainfall_mm is not None:
            raise ValueError("Basins-only preparation cannot include rainfall.")
        if self.rainfall_mm is None and any(
            value is not None
            for value in (
                self.infiltration_fraction,
                self.infiltration_capacity_mm_h,
                self.rainfall_duration_hours,
            )
        ):
            raise ValueError("Infiltration and duration options require rainfall.")
        if self.infiltration_fraction is not None and not 0 <= self.infiltration_fraction <= 1:
            raise ValueError("Infiltration fraction must be between zero and one.")
        if self.minimum_depression_depth_m < 0 or self.minimum_basin_area_m2 < 0:
            raise ValueError("Basin depth and area thresholds cannot be negative.")
        if not math.isfinite(self.center_lon) or not -180 <= self.center_lon <= 180:
            raise ValueError("Center longitude must be between -180 and 180 degrees.")
        if not math.isfinite(self.center_lat) or not -90 <= self.center_lat <= 90:
            raise ValueError("Center latitude must be between -90 and 90 degrees.")
        for direction in ("west", "east", "north", "south"):
            extent = getattr(self, f"{direction}_km")
            buffer = getattr(self, f"buffer_{direction}_km")
            if not math.isfinite(extent) or extent <= 0:
                raise ValueError(f"{direction.title()} extent must be finite and positive.")
            if not math.isfinite(buffer) or buffer < 0:
                raise ValueError(f"{direction.title()} buffer must be finite and non-negative.")
        if self.infiltration_capacity_mm_h is not None:
            if self.infiltration_capacity_mm_h < 0:
                raise ValueError("Infiltration capacity cannot be negative.")
            if self.rainfall_duration_hours is None or self.rainfall_duration_hours <= 0:
                raise ValueError("Infiltration capacity requires a positive storm duration.")
        elif self.rainfall_duration_hours is not None:
            raise ValueError("Storm duration requires an infiltration capacity.")


@dataclass(frozen=True)
class PySwmmConfig:
    """Inputs and options for one location-independent conceptual SWMM run."""

    effective_rainfall_mm: float = 100.0
    duration_hours: float = 1.0
    drain_hours: float = 2.0
    report_interval_minutes: int = 5
    minimum_storage_basin_area_m2: float = 100.0
    spill_channel_width_m: float = 1.0
    spill_channel_roughness: float = 0.03
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Validate settings independent of the SWMM engine."""
        if (
            not math.isfinite(self.effective_rainfall_mm)
            or self.effective_rainfall_mm < 0
        ):
            raise ValueError("Effective rainfall must be finite and non-negative.")
        if not math.isfinite(self.duration_hours) or self.duration_hours <= 0:
            raise ValueError("Storm duration must be finite and positive.")
        if not math.isfinite(self.drain_hours) or self.drain_hours < 0:
            raise ValueError("Drain time must be finite and non-negative.")
        if (
            self.report_interval_minutes <= 0
            or 60 % self.report_interval_minutes
            or int(self.report_interval_minutes) != self.report_interval_minutes
        ):
            raise ValueError(
                "Report interval must be a positive whole number of minutes"
                " that divides 60 evenly."
            )
        if self.spill_channel_width_m <= 0 or not math.isfinite(
            self.spill_channel_width_m
        ):
            raise ValueError("Spill-channel width must be finite and positive.")
        if self.spill_channel_roughness <= 0 or not math.isfinite(
            self.spill_channel_roughness
        ):
            raise ValueError("Spill-channel roughness must be finite and positive.")
        if self.minimum_storage_basin_area_m2 < 0:
            raise ValueError("Minimum storage basin area cannot be negative.")
