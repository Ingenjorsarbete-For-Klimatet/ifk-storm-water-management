"""Compatibility command for the Enköping terrain-results plot.

The plotting implementation is in ``storm_water_management.utils_plots``.
Prefer ``python examples/plot_enkoping.py terrain``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from storm_water_management.utils_plots import main as plot_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Run the terrain-results plot with the legacy CLI."""
    plot_main("terrain", argv)


if __name__ == "__main__":
    main()
