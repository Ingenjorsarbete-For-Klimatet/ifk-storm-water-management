"""Diagnose basin-network problems from rasters that a previous run already wrote.

Reads <prefix>_buffered_dem / _depression_depth / _d8_pointer /
_d8_contributing_area from the results directory (no Whitebox re-run), rebuilds
the basin network and reports why spill links are dropped or spill paths end
unresolved. It also writes a CSV of problem basins with map coordinates for QGIS.

Example (from the repository root):

    python diagnose_basin_network.py --results-dir results/enkoping --prefix enkoping
"""

from __future__ import annotations

import argparse
import csv
import importlib
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio.transform


def _config_defaults(module_name: str) -> dict:
    """Read default threshold values from AnalysisConfig, if it defines them."""
    import dataclasses

    package = module_name.rsplit(".", 1)[0]
    try:
        config = importlib.import_module(f"{package}.config")
        return {
            f.name: f.default
            for f in dataclasses.fields(config.AnalysisConfig)
            if f.default is not dataclasses.MISSING
        }
    except Exception:  # config unavailable: fall back to the built-in defaults
        return {}


def _xy(transform, flat_index: int, width: int) -> tuple[float, float]:
    if flat_index < 0:
        return float("nan"), float("nan")
    row, col = divmod(int(flat_index), width)
    x, y = rasterio.transform.xy(transform, row, col, offset="center")
    return float(x), float(y)


def _quantiles(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return "  ".join(
        f"p{p}={v:.3g}" for p, v in zip((5, 25, 50, 75, 95), q, strict=True)
    )


def main() -> None:
    """Rebuild the basin network from result rasters and report problem basins."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--module", default="storm_water_management.storm_scenario")
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="default: AnalysisConfig.minimum_depression_depth_m",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help="default: AnalysisConfig.minimum_basin_area_m2",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    mod = importlib.import_module(args.module)
    config_defaults = _config_defaults(args.module)
    min_depth = (
        args.min_depth
        if args.min_depth is not None
        else config_defaults.get("minimum_depression_depth_m", 0.02)
    )
    min_area = (
        args.min_area
        if args.min_area is not None
        else config_defaults.get("minimum_basin_area_m2", 25.0)
    )
    print(f"Thresholds: minimum depth {min_depth:g} m, minimum area {min_area:g} m2")
    folder, prefix = args.results_dir, args.prefix
    data = mod._read_inputs(
        folder / f"{prefix}_buffered_dem.tif",
        folder / f"{prefix}_buffered_depression_depth.tif",
        folder / f"{prefix}_buffered_d8_pointer.tif",
        folder / f"{prefix}_buffered_d8_contributing_area.tif",
        "Diagnostic",
    )
    height, width = data.dem.shape
    print(f"Raster {height} x {width}, cell area {data.cell_area:g} m2")

    valid = mod._valid_mask(data.dem, data.depression, data.pointer, data.accumulation)
    pointer_zero = valid & (data.pointer == 0)
    border = np.zeros(valid.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    print("\n== D8 pointer ==")
    print(f"valid cells: {int(valid.sum()):,}")
    print(
        f"pointer == 0 inside the raster: {int((pointer_zero & ~border).sum()):,} "
        f"(on the border: {int((pointer_zero & border).sum()):,})"
    )
    print(
        "depth > epsilon cells: "
        f"{int((valid & (data.depression > mod.LABEL_EPSILON_M)).sum()):,}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = mod._build_network(
            data.dem,
            data.depression,
            data.pointer,
            data.accumulation,
            data.cell_area,
            min_depth,
            min_area,
            mod.LABEL_EPSILON_M,
            False,
            None,
        )
    n = net.basin_count
    print(f"\n== Basins: {n:,} ==")
    statuses = Counter(mod._status(net, b) for b in range(1, n + 1))
    for key, value in statuses.most_common():
        print(f"  {key:32s} {value:8,d}  ({value / max(n, 1):.1%})")
    reasons = Counter(net.trace_reason[1:])
    print("\nTrace end reasons:")
    for key, value in reasons.most_common():
        print(f"  {key:32s} {value:8,d}")
    print("\nBasin size (cells):", _quantiles(net.counts[1:].astype(float)))
    print(
        "Outlet area / own area:",
        _quantiles(
            net.outlet_area[1:] / np.maximum(net.counts[1:] * data.cell_area, 1e-9)
        ),
    )

    problems: list[dict] = []
    flat_dem = data.dem.reshape(-1)
    flat_ptr = data.pointer.reshape(-1)
    flat_labels = net.labels.reshape(-1)

    unresolved = [
        b for b in range(1, n + 1) if mod._status(net, b) == mod.STATUS_UNRESOLVED
    ]
    print(f"\n== Unresolved spill paths: {len(unresolved):,} ==")
    if unresolved:
        ends = net.trace_end[unresolved]
        end_ptr = flat_ptr[ends]
        print(
            "pointer value at trace end:",
            dict(Counter(int(v) if np.isfinite(v) else -1 for v in end_ptr)),
        )
        print(
            "trace end inside a retained basin:",
            f"{int((flat_labels[ends] > 0).sum()):,}",
        )
        print(
            "steps before ending:",
            _quantiles(net.trace_steps[unresolved].astype(float)),
        )
        print(
            "dem at trace end - spill level:",
            _quantiles(flat_dem[ends] - net.spill[unresolved]),
        )
    for b in unresolved:
        ox, oy = _xy(data.transform, int(net.inside_outlet[b]), width)
        ex, ey = _xy(data.transform, int(net.trace_end[b]), width)
        problems.append(
            dict(
                basin_id=b,
                kind="unresolved",
                reason=net.trace_reason[b],
                cell_count=int(net.counts[b]),
                outlet_x=ox,
                outlet_y=oy,
                end_x=ex,
                end_y=ey,
                steps=int(net.trace_steps[b]),
                sender_area_m2=float(net.outlet_area[b]),
                receiver_id=0,
                receiver_area_m2=float("nan"),
            )
        )
    for row in problems[: args.samples]:
        print(
            "  e.g.",
            {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()},
        )

    dropped = np.flatnonzero(net.link_dropped)
    print(f"\n== Dropped links: {dropped.size:,} ==")
    if dropped.size:
        receivers = net.dropped_receiver[dropped]
        sender_area = net.outlet_area[dropped]
        receiver_area = net.outlet_area[receivers]
        ratio = receiver_area / np.maximum(sender_area, 1e-9)
        print("receiver area / sender area:", _quantiles(ratio))
        print(
            f"  exactly equal (|diff| < 1e-6 m2): {int((np.abs(receiver_area - sender_area) < 1e-6).sum()):,}"
        )
        print(
            f"  receiver smaller than sender:     {int((receiver_area < sender_area - 1e-6).sum()):,}"
        )
        print(
            f"  receiver outlet area equals its own basin area: "
            f"{int((receiver_area <= net.counts[receivers] * data.cell_area + 1e-6).sum()):,}"
        )
        print("sender area (m2):", _quantiles(sender_area))
        print("receiver area (m2):", _quantiles(receiver_area))
        reasons_dropped = Counter(net.trace_reason[b] for b in dropped)
        print("trace end reasons of dropped:", dict(reasons_dropped))
        for b, r in zip(dropped, receivers, strict=True):
            ox, oy = _xy(data.transform, int(net.inside_outlet[b]), width)
            rx, ry = _xy(data.transform, int(net.inside_outlet[r]), width)
            problems.append(
                dict(
                    basin_id=int(b),
                    kind="dropped",
                    reason=net.trace_reason[b],
                    cell_count=int(net.counts[b]),
                    outlet_x=ox,
                    outlet_y=oy,
                    end_x=rx,
                    end_y=ry,
                    steps=int(net.trace_steps[b]),
                    sender_area_m2=float(net.outlet_area[b]),
                    receiver_id=int(r),
                    receiver_area_m2=float(net.outlet_area[r]),
                )
            )
        print("examples (largest sender area first):")
        for b in dropped[np.argsort(-net.outlet_area[dropped])][: args.samples]:
            row = next(
                p
                for p in problems
                if p["basin_id"] == int(b) and p["kind"] == "dropped"
            )
            print(
                "  ",
                {
                    k: (round(v, 1) if isinstance(v, float) else v)
                    for k, v in row.items()
                },
            )

    out = args.output_csv or folder / f"{prefix}_basin_network_problems.csv"
    if problems:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(problems[0]))
            writer.writeheader()
            writer.writerows(problems)
        print(f"\nProblem basins with SWEREF99 TM coordinates: {out}")


if __name__ == "__main__":
    main()
