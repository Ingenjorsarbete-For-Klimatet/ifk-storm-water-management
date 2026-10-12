"""Download Lantmäteriet's 1 m elevation tiles around Enköping.

By default, the search covers a 10 km radius around Enköping town. Use
``--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT`` to specify a different area, such
as a municipality-wide bounding box. Set ``LANTMATERIET_USER`` and
``LANTMATERIET_PASSWORD`` in the environment for account authentication.
Files are saved under ``data/elevation/enkoping``.

Run from the repository root with ``python examples/download_enkoping_elevation.py``.
"""

import argparse
import base64
import json
import math
import os
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

import rasterio

STAC_URL = "https://api.lantmateriet.se/stac-hojd/v1/search"
COLLECTION = "mhm-66_6"
USER = os.environ.get("LANTMATERIET_USER")
PASS = os.environ.get("LANTMATERIET_PASSWORD")
ENKOPING_LON = 17.0778
ENKOPING_LAT = 59.6356
DEFAULT_OUTPUT = Path("data/elevation/enkoping")
REQUEST_TIMEOUT_SECONDS = 200
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def bbox_for_radius(radius_km: float) -> tuple[float, float, float, float]:
    """Return an approximate WGS84 bounding box around Enköping town."""
    if radius_km <= 0:
        raise ValueError("The search radius must be greater than zero.")

    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * math.cos(math.radians(ENKOPING_LAT)))
    return (
        ENKOPING_LON - lon_delta,
        ENKOPING_LAT - lat_delta,
        ENKOPING_LON + lon_delta,
        ENKOPING_LAT + lat_delta,
    )


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Validate a WGS84 STAC bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError("Longitude bounds must satisfy -180 <= min < max <= 180.")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError("Latitude bounds must satisfy -90 <= min < max <= 90.")


def get_json(url: str) -> dict:
    """Request and decode a JSON response from the STAC API."""
    parsed_url = urlsplit(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "api.lantmateriet.se":
        raise ValueError("Refusing a STAC request outside Lantmäteriet HTTPS.")

    request = Request(  # noqa: S310
        url, headers={"Accept": "application/geo+json, application/json"}
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
        return json.load(response)


def find_tiles(bbox: tuple[float, float, float, float], collection: str) -> list[dict]:
    """Search the collection and follow STAC pagination links."""
    query = urlencode(
        {
            "collections": collection,
            "bbox": ",".join(f"{coordinate:.6f}" for coordinate in bbox),
            "limit": 100,
        }
    )
    page_url = f"{STAC_URL}?{query}"
    items_by_id: dict[str, dict] = {}

    while page_url:
        page = get_json(page_url)
        for item in page.get("features", []):
            item_id = item.get("id") or item.get("assets", {}).get("data", {}).get(
                "href", ""
            )
            items_by_id[item_id] = item

        next_link = next(
            (link for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )
        page_url = urljoin(page_url, next_link["href"]) if next_link else ""

    return list(items_by_id.values())


def make_auth_header() -> str:
    """Build an HTTP Basic authorization value from environment variables."""
    username = USER
    password = PASS
    if bool(username) != bool(password):
        raise ValueError(
            "Set both LANTMATERIET_USER and LANTMATERIET_PASSWORD."
        )
    if not username or not password:
        raise ValueError(
            "Lantmäteriet download credentials are required. Set both "
            "LANTMATERIET_USER and LANTMATERIET_PASSWORD in the environment."
        )

    encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {encoded}"


def is_valid_geotiff(path: Path) -> bool:
    """Return whether a file can be opened as a non-empty raster."""
    try:
        with rasterio.open(path) as dataset:
            return dataset.width > 0 and dataset.height > 0
    except (rasterio.errors.RasterioError, OSError):
        return False


def download_tile(href: str, output_dir: Path, auth_header: str) -> Path:
    """Download one tile atomically, keeping valid existing files."""
    parsed_href = urlsplit(href)
    if parsed_href.scheme != "https":
        raise ValueError("Refusing to download an elevation asset over non-HTTPS.")

    filename = unquote(PurePosixPath(parsed_href.path).name)
    if not filename:
        raise ValueError("A STAC data asset did not include a filename.")

    destination = output_dir / filename
    if destination.exists() and is_valid_geotiff(destination):
        return destination
    if destination.exists():
        destination.unlink()

    headers = {"User-Agent": "ifk-storm-water-management elevation downloader"}
    hostname = parsed_href.hostname or ""
    is_lantmateriet_host = hostname == "lantmateriet.se" or hostname.endswith(
        ".lantmateriet.se"
    )
    if is_lantmateriet_host:
        headers["Authorization"] = auth_header
    request = Request(href, headers=headers)  # noqa: S310
    partial_file = destination.with_name(destination.name + ".part")

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            with partial_file.open("wb") as output:
                while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
        if not is_valid_geotiff(partial_file):
            raise ValueError(f"Downloaded asset {filename} is not a valid GeoTIFF.")
        partial_file.replace(destination)
    except (HTTPError, URLError) as error:
        partial_file.unlink(missing_ok=True)
        if isinstance(error, HTTPError) and error.code == 401:
            raise RuntimeError(
                "Lantmäteriet rejected the download. Check your account access and "
                "the LANTMATERIET_USER / LANTMATERIET_PASSWORD environment variables."
            ) from error
        raise RuntimeError(f"Could not download {filename}: {error}") from error
    except Exception:
        partial_file.unlink(missing_ok=True)
        raise

    return destination


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radius-km",
        type=float,
        default=10.0,
        help="Search radius around Enköping town (default: 10 km; ignored with --bbox).",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Explicit WGS84 bounding box; use this to cover a larger area/municipality.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for downloaded tiles (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"Lantmäteriet STAC collection (default: {COLLECTION}).",
    )
    args = parser.parse_args()
    args.bbox = tuple(args.bbox) if args.bbox else bbox_for_radius(args.radius_km)
    try:
        validate_bbox(args.bbox)
        args.auth_header = make_auth_header()
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    """Find and download all elevation tiles intersecting the selected area."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching collection {args.collection} for bbox {args.bbox} ...")
    try:
        items = find_tiles(args.bbox, args.collection)
        if not items:
            print("No elevation tiles found for this area.")
            return

        print(f"Found {len(items)} tiles. Saving to {args.output_dir}.")
        downloaded = 0
        for item in items:
            asset = item.get("assets", {}).get("data")
            if not asset or not asset.get("href"):
                print(f"Skipping {item.get('id', '<unknown item>')}: no data asset.")
                continue

            path = download_tile(asset["href"], args.output_dir, args.auth_header)
            print(f"Ready: {path.name}")
            downloaded += 1

        print(f"Finished. {downloaded} tile(s) are available in {args.output_dir}.")
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Lantmäteriet STAC request failed: {error}") from error
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
