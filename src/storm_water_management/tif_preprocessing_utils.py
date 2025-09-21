"""Utils."""

import os

import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.plot import show


def concat_tif(files: list, folder: str, output_filname: str) -> None:
    """Concat tif files.

    Args:
        files: list of files
        folder: file folder
        output_filname: name of merged output file
    """
    src_files = [rasterio.open(os.path.join(folder, f)) for f in files]
    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )

    with rasterio.open(output_filname, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()

    plot_merge = False
    if plot_merge:
        with rasterio.open(output_filname) as src:
            fig, ax = plt.subplots(figsize=(8, 6))
            show(src, ax=ax, cmap="viridis")
            plt.show()

        src.close()


if __name__ == "__main__":
    path = "/home/chris/repos/storm_temp/data/1m"
    filenames = ["63950_3150_25.tif", "63975_3150_25.tif"]
    output_name = "meged.tif"
    concat_tif(filenames, path, output_name)
