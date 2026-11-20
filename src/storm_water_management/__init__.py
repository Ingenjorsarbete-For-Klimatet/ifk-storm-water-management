"""Terrain-based storm-water screening from elevation data.

The package downloads elevation tiles, derives depression basins and their
spill network from a DEM, screens rainfall storage and spill volumes, and can
build a conceptual SWMM model of the basin network. The public entry points
are re-exported here; the modules behind them are:

* ``config``: input dataclasses for each workflow.
* ``elevation_download``: DEM tile download from Lantmateriet's STAC API.
* ``terrain``: raster and terrain processing steps.
* ``basin_network``: depression labelling, spill topology, storage/spill
  simulation, and basin raster/CSV outputs.
* ``terrain_analysis``: the end-to-end analysis workflow.
* ``pyswmm_model``: conceptual SWMM model construction and simulation.
* ``plotting``: result figures.
* ``depth_polygons``: depth-class polygon vector export.
"""

from __future__ import annotations

from storm_water_management.config import AnalysisConfig, DownloadConfig, PySwmmConfig
from storm_water_management.depth_polygons import write_depth_class_polygons
from storm_water_management.elevation_download import download_elevation_tiles
from storm_water_management.plotting import create_pyswmm_plot, create_terrain_plot
from storm_water_management.pyswmm_model import run_pyswmm_analysis
from storm_water_management.terrain_analysis import run_terrain_analysis

__all__ = [
    "AnalysisConfig",
    "DownloadConfig",
    "PySwmmConfig",
    "create_pyswmm_plot",
    "create_terrain_plot",
    "download_elevation_tiles",
    "run_pyswmm_analysis",
    "run_terrain_analysis",
    "write_depth_class_polygons",
]
