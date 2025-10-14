"""Utils for preprocessing of tif files."""

import argparse
import os

import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.plot import show


def concat_tif_in_folder(folder: str, plot_merge: bool = True, output_filename: str = "") -> None:
    """Concat tif files in folder.

    Args:
        folder: folder including tif files to merge
        plot_merge: plot merge (True/False - default True)
        output_filename: Filename of merged file (optional)
    """
    files = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    if not files:
        raise ValueError(f"No .tif-files in: {folder}")

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

    folder_name = os.path.basename(os.path.normpath(folder))
    if len(output_filename)==0:
        output_filename = os.path.join(folder, f"{folder_name}.tif")

    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files:
        src.close()

    if plot_merge:
        with rasterio.open(output_filename) as src:
            fig, ax = plt.subplots(figsize=(8, 6))
            show(src, ax=ax, cmap="viridis")
            plt.show()

        src.close()


def parse_and_run() -> None:
    """Parse and run."""
    parser = argparse.ArgumentParser(description="Concat all files in folder to 1 tif")
    parser.add_argument("folder", help="Folder including tif files")
    args = parser.parse_args()
    concat_tif_in_folder(args.folder)


if __name__ == "__main__":
    parse_and_run()
