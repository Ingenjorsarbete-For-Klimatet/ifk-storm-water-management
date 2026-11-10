"""Download elevation tiles from Lantmäteriet's STAC API."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urljoin, urlsplit
from urllib.request import Request, urlopen

import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAC_URL = "https://api.lantmateriet.se/stac-hojd/v1/search"
COLLECTION = "dtm-cog"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "elevation"
REQUEST_TIMEOUT_SECONDS = 200
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def bbox_for_radius(
    radius_km: float, longitude: float, latitude: float
) -> tuple[float, float, float, float]:
    """Return an approximate WGS84 bounding box around a coordinate."""
    if not math.isfinite(radius_km) or radius_km <= 0:
        raise ValueError("The search radius must be greater than zero.")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Center longitude must be between -180 and 180 degrees.")
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Center latitude must be between -90 and 90 degrees.")

    lat_delta = radius_km / 111.32
    longitude_scale = 111.32 * math.cos(math.radians(latitude))
    if abs(longitude_scale) < 1e-8:
        raise ValueError("A radius-based search is not supported at the poles; use a bounding box.")
    lon_delta = radius_km / longitude_scale
    return (
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    )


def location_slug(name: str) -> str:
    """Convert a display name to a safe, portable output-directory name."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    if not slug:
        raise ValueError("The location name must contain at least one letter or number.")
    return slug


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

    request = Request(url, headers={"Accept": "application/geo+json, application/json"})  # noqa: S310
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
            item_id = item.get("id") or item.get("assets", {}).get("data", {}).get("href", "")
            items_by_id[item_id] = item
        next_link = next(
            (link for link in page.get("links", []) if link.get("rel") == "next"),
            None,
        )
        page_url = urljoin(page_url, next_link["href"]) if next_link else ""
    return list(items_by_id.values())


def make_auth_header() -> str:
    """Build an HTTP Basic authorization value from environment variables."""
    username = os.environ.get("LANTMATERIET_USER")
    password = os.environ.get("LANTMATERIET_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("Set both LANTMATERIET_USER and LANTMATERIET_PASSWORD.")
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
    if hostname == "lantmateriet.se" or hostname.endswith(".lantmateriet.se"):
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
                "Lantmäteriet rejected the download. Check account access and the "
                "LANTMATERIET_USER / LANTMATERIET_PASSWORD environment variables."
            ) from error
        raise RuntimeError(f"Could not download {filename}: {error}") from error
    except Exception:
        partial_file.unlink(missing_ok=True)
        raise
    return destination


def download_elevation_tiles(
    *,
    name: str,
    center_lon: float | None = None,
    center_lat: float | None = None,
    radius_km: float = 10.0,
    bbox: tuple[float, float, float, float] | None = None,
    output_dir: Path | None = None,
    collection: str = COLLECTION,
    auth_header: str | None = None,
) -> list[Path]:
    """Find and download elevation tiles intersecting a WGS84 area."""
    location_slug(name)
    if bbox is None:
        if center_lon is None or center_lat is None:
            raise ValueError("Supply either bbox or both center longitude and latitude.")
        selected_bbox = bbox_for_radius(radius_km, center_lon, center_lat)
    else:
        selected_bbox = bbox
    validate_bbox(selected_bbox)
    destination_dir = output_dir or DEFAULT_OUTPUT_ROOT / location_slug(name)
    authorization = auth_header or make_auth_header()
    destination_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for {name} in collection {collection} within bbox {selected_bbox} ...")
    items = find_tiles(selected_bbox, collection)
    if not items:
        print("No elevation tiles found for this area.")
        return []

    print(f"Found {len(items)} tiles. Saving to {destination_dir}.")
    downloaded: list[Path] = []
    for item in items:
        asset = item.get("assets", {}).get("data")
        if not asset or not asset.get("href"):
            print(f"Skipping {item.get('id', '<unknown item>')}: no data asset.")
            continue
        path = download_tile(asset["href"], destination_dir, authorization)
        print(f"Ready: {path.name}")
        downloaded.append(path)
    print(f"Finished. {len(downloaded)} tile(s) are available in {destination_dir}.")
    return downloaded
