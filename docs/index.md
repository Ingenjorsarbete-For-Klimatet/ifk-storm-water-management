# Welcome to ifk-storm-water-management

## Data

The elevation data used in this repository is downloaded from **Lantmäteriet**.
It consists of **raster elevation data** (preferably 1 m resolution) provided as
**.tif files**.

## Pre-processing

If you need to analyze a larger area than a single **.tif** file covers,
the files must first be **merged**.

To run script from terminal:

<pre>
python src/storm_water_management/tif_preprocessing_utils.py "Folder including
tif-files to merge"
</pre>

Or import and run in Python

<pre>
import tif_preprocessing_utils

concat_tif_in_folder(folder)
</pre>

## Analysis

To perform the main analysis, run:

<pre>
python src/storm_water_management/main.py "Path to file, e.g., <data/file.tif>"
</pre>

Or import and run in Python

<pre>
Import main

main(filename)
</pre>
