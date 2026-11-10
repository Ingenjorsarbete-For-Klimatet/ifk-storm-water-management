# Storm Water Management

Tools for terrain-based depression analysis, rainfall screening, and conceptual
SWMM simulations using elevation rasters.

## Elevation data

Elevation tiles can be downloaded from Lantmäteriet with
`storm_water_management.elevation_download.download_elevation_tiles()`. Configure
the location and area with `DownloadConfig` from
`storm_water_management.config`. Supply a location name and either a WGS84
bounding box or center coordinates. Set
`LANTMATERIET_USER` and `LANTMATERIET_PASSWORD` in the environment when required
by the data service.

## Terrain analysis

`AnalysisConfig` and `PySwmmConfig` are defined in
[config.py](../src/storm_water_management/config.py). The
`run_terrain_analysis()` function is defined in
[storm_scenario.py](../src/storm_water_management/storm_scenario.py). Supply the
input/output directories, center coordinates, and an output prefix explicitly;
the package does not choose a geographic location:

```python
from pathlib import Path

from storm_water_management.config import AnalysisConfig
from storm_water_management.storm_scenario import run_terrain_analysis

config = AnalysisConfig(
    input_dir=Path("data/elevation/tiles"),
    output_dir=Path("results/terrain"),
    center_lon=12.0,
    center_lat=55.0,
    output_prefix="study_area",
    west_km=2.0,
    east_km=2.0,
    north_km=2.0,
    south_km=2.0,
)
run_terrain_analysis(config)
```

To add a static rainfall screening estimate, set `rainfall_mm`. Infiltration
can be specified as a fraction or as a constant capacity together with
`rainfall_duration_hours`. The generated rasters and basin summaries are written
to the configured output directory. This is a DEM/D8 screening method, not a
calibrated hydraulic flood model.

## Conceptual SWMM analysis

`PySwmmConfig` is defined in
[config.py](../src/storm_water_management/config.py), and
`run_pyswmm_analysis()` is defined in
[utils_pyswmm.py](../src/storm_water_management/utils_pyswmm.py). Provide the
summary, DEM, basin-ID, and output paths explicitly:

```python
from pathlib import Path

from storm_water_management.config import PySwmmConfig
from storm_water_management.utils_pyswmm import run_pyswmm_analysis

run_pyswmm_analysis(
    PySwmmConfig(effective_rainfall_mm=100, duration_hours=6),
    basin_summary_path=Path("results/terrain/study_area_buffered_basin_summary.csv"),
    dem_path=Path("results/terrain/study_area_buffered_dem.tif"),
    basin_ids_path=Path("results/terrain/study_area_buffered_depression_basin_id.tif"),
    output_dir=Path("results/swmm"),
)
```

Install the optional dependency with `pip install -e '.[swmm]'`. The conceptual
model uses DEM-derived storage basins, uniform net rainfall, and assumed
routing parameters; its results are exploratory and not infrastructure-grade
flood predictions.

## Visualization

Plot functions are available in
[utils_plots.py](../src/storm_water_management/utils_plots.py). They take input
raster/vector paths and output image paths directly, so callers can organize
results for any study area.
