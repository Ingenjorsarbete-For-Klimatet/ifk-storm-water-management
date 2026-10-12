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

## Enköping rainfall scenario (screening)

The Enköping example optionally estimates runoff from an event rainfall depth,
routes the event volume through connected DEM-derived depression basins, stores
water up to each estimated spill level, and sends excess volume downstream
along D8 flow paths. For example, to run a 10 mm event with the default 50%
infiltration assumption:

```sh
python examples/analyze_enkoping.py --west-km 1.5 --east-km 1.5 \
  --north-km 1 --south-km 1.5 --rainfall-mm 10
```

Use `--west-km`, `--east-km`, `--north-km`, and `--south-km` to change the
result area. Use the four `--buffer-*-km` options to give upstream terrain room
to drain into it. The default basin filters exclude components shallower than
2 cm or smaller than 25 m²; adjust with `--minimum-depression-depth-m` and
`--minimum-basin-area-m2` if needed.

The rainfall-conditioned depth raster is
`results/enkoping/enkoping_estimated_ponding_depth.tif`. Per-basin input,
storage, spill, outlet, and downstream-basin details are written to
`results/enkoping/enkoping_buffered_basin_summary.csv`; basin IDs are in
`results/enkoping/enkoping_depression_basin_id.tif`. Recreate the visualization
with `python examples/plot_enkoping.py terrain`.

This is a static DEM-only screening estimate, not a hydraulic flood model. It
uses uniform effective rainfall and D8 routing; it does not include storm
timing, spatially varying infiltration, culverts, sewers, buildings, or
calibration. Depressions touching NoData or the analysis edge are treated as
censored/open. Results depend on the basin size/depth filters and buffer size.
The current method labels 8-connected fill zones, not a complete nested
depression hierarchy, so adjacent or nested hollows can be grouped together.

## Enköping PySWMM scenario (conceptual)

The standalone `examples/analyze_enkoping_pyswmm.py` script uses the existing
basin summary plus its matching buffered DEM and basin-ID raster to create
SWMM storage curves from DEM elevations, connect storage nodes using the
existing spill graph, and run a time-varying event simulation. Install the
optional dependency with `pip install -e '.[swmm]'`, then run
`python examples/analyze_enkoping_pyswmm.py` from the repository root. Outputs
are written to `results/enkoping/pyswmm/`; use `--effective-rainfall-mm` and
`--duration-hours` to test a different uniform effective-rainfall scenario.
By default, only depressions of at least 1,000 m² are represented as storage
nodes. Smaller basins are collapsed into the next modeled basin or routed to a
boundary outfall, while their incremental runoff area is conserved. Adjust
`--minimum-storage-basin-area-m2` to change that tradeoff. The output includes
node/link time-series CSVs, a per-basin peak summary, the generated SWMM input,
and model assumptions/continuity information.

Create a PySWMM visualization of basin peak depth, node flooding, aggregate
boundary outflow, and stored volume with
`python examples/plot_enkoping.py pyswmm`. The default PNG is
`results/enkoping/pyswmm/enkoping_pyswmm_results.png`.

No surveyed storm-drain network, calibrated surface roughness, or design
hyetograph is available in the repository. The script therefore simulates
DEM-derived surface depressions only, uses a uniform storm and assumed
overland-flow/weir parameters, and treats the CSV rainfall as net rainfall
(without subtracting infiltration again). Its output is an exploratory dynamic
screening scenario, not a calibrated or infrastructure-grade flood prediction.
