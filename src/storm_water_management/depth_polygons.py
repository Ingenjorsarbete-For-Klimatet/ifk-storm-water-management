"""Convert water-depth rasters to depth-class polygon vectors."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape

DEPTH_CLASS_BINS_M = (0.1, 0.2, 0.5, 1.0, 1000.0)
DEPTH_CLASS_LABELS = ("0.1-0.2 m", "0.2-0.5 m", "0.5-1 m", "1 <")


def write_depth_class_polygons(
    raster_path: Path,
    output_format: str = "GeoJSON",
    apply_dissolve: bool = False,
    plot_polygons: bool = False,
) -> None:
    """Classify a depth raster and write one polygon per depth class.

    Args:
        raster_path: water-depth raster (metres; values <= 0 are ignored).
        output_format: vector driver, either "GeoJSON" or "FlatGeobuf".
        apply_dissolve: merge polygons of the same depth class.
        plot_polygons: show a quick class-coloured map of the polygons.
    """
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as source:
        depths = source.read(1).astype(float)
        transform = source.transform
        crs = source.crs

    classified = np.digitize(depths, DEPTH_CLASS_BINS_M, right=False).astype("int16")
    wet_mask = depths > 0

    geoms = []
    class_names = []
    for geometry, class_index in features.shapes(
        classified, mask=wet_mask, transform=transform
    ):
        class_index = int(class_index)
        if 0 < class_index <= len(DEPTH_CLASS_LABELS):
            geoms.append(shape(geometry))
            class_names.append(DEPTH_CLASS_LABELS[class_index - 1])

    gdf = gpd.GeoDataFrame({"depth_class": class_names}, geometry=geoms, crs=crs)

    if apply_dissolve:
        gdf = gdf.dissolve(by="depth_class")
        gdf = gdf[gdf.geometry.notna()]

    file_extension = "fgb" if output_format == "FlatGeobuf" else "geojson"
    output_path = raster_path.with_name(f"{raster_path.stem}_polygons.{file_extension}")
    gdf = gdf.to_crs(epsg=4326)
    gdf.to_file(output_path, driver=output_format)

    if plot_polygons:
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column="depth_class", cmap="viridis", legend=True, ax=ax)
        plt.title("Vattendjup")
        plt.show()
