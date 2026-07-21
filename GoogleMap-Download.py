"""
GoogleMap-Download.py

Download Google Maps Satellite imagery covering an area of interest (AOI),
stitch the tiles into a single georeferenced GoogleMap-Images.tif, and
build an external pyramid (.ovr) for fast display in GIS software.

Usage:
    python GoogleMap-Download.py

Configure the AOI and ZOOM constants below before running. The default
AOI is the polygon supplied by the user (Bangkok, TH).

Requirements (already installed in .venv):
    rasterio, requests, pillow, tqdm

Note: Google Maps tiles are provided for the Google Maps/Earth apps.
Automated downloading may be against Google's Terms of Service - use
this only for personal, non-commercial, or otherwise authorized purposes.
"""

import ctypes
import glob
import io
import math
import os
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# AOI polygon (lon, lat) - taken from the Earth Engine geometry provided.
AOI_POLYGON = [
    (100.56346324317587, 13.74706886700266),
    (100.56346324317587, 13.743337940191722),
    (100.56627419822348, 13.743337940191722),
    (100.56627419822348, 13.74706886700266),
]

ZOOM = 19               # Google tile zoom level (0-21). Higher = more detail.
TILE_SIZE = 256         # Google tiles are 256x256 px.
TILE_SERVERS = [        # round-robin across mirrors to spread load
    "https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "https://mt3.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
]
OUTPUT_TIF = "GoogleMap-Images.tif"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 15
OVERVIEW_FACTORS = [2, 4, 8, 16, 32]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# --------------------------------------------------------------------------
# Web Mercator / slippy-map tile math
# --------------------------------------------------------------------------

WEB_MERCATOR_R = 6378137.0  # Earth radius used by EPSG:3857


def lonlat_to_tile(lon, lat, zoom):
    """Return fractional (x, y) tile coordinates for lon/lat at zoom."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_mercator(x, y, zoom):
    """Return EPSG:3857 (meters) coordinates of a tile's top-left corner."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)

    mx = math.radians(lon_deg) * WEB_MERCATOR_R
    my = math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2)) * WEB_MERCATOR_R
    return mx, my


def bounding_tiles(polygon, zoom):
    """Return the integer tile x/y range covering the polygon's bbox."""
    xs = [lonlat_to_tile(lon, lat, zoom)[0] for lon, lat in polygon]
    ys = [lonlat_to_tile(lon, lat, zoom)[1] for lon, lat in polygon]
    x_min, x_max = math.floor(min(xs)), math.ceil(max(xs))
    y_min, y_max = math.floor(min(ys)), math.ceil(max(ys))
    return x_min, x_max, y_min, y_max


# --------------------------------------------------------------------------
# Tile download
# --------------------------------------------------------------------------

def download_tile(session, x, y, zoom):
    """Download a single tile, retrying on transient failures."""
    for attempt in range(MAX_RETRIES):
        url = TILE_SERVERS[(x + y) % len(TILE_SERVERS)].format(x=x, y=y, z=zoom)
        try:
            resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download tile z={zoom} x={x} y={y}")


def download_mosaic(x_min, x_max, y_min, y_max, zoom):
    """Download all tiles in range and stitch them into one RGB array."""
    n_cols = x_max - x_min
    n_rows = y_max - y_min
    mosaic = Image.new("RGB", (n_cols * TILE_SIZE, n_rows * TILE_SIZE))

    with requests.Session() as session:
        tiles = [(x, y) for y in range(y_min, y_max) for x in range(x_min, x_max)]
        for x, y in tqdm(tiles, desc="Downloading tiles"):
            tile_img = download_tile(session, x, y, zoom)
            mosaic.paste(tile_img, ((x - x_min) * TILE_SIZE, (y - y_min) * TILE_SIZE))

    return mosaic


# --------------------------------------------------------------------------
# GeoTIFF writing + external pyramid
# --------------------------------------------------------------------------

def save_geotiff(mosaic, x_min, y_min, zoom, out_path):
    """Write the mosaic to a georeferenced GeoTIFF in EPSG:3857."""
    # Meters-per-pixel at this zoom (Web Mercator tile is 2*pi*R meters wide / 2^zoom).
    tile_span_m = 2 * math.pi * WEB_MERCATOR_R / (2 ** zoom)
    px_size = tile_span_m / TILE_SIZE

    origin_x, origin_y = tile_to_mercator(x_min, y_min, zoom)
    transform = from_origin(origin_x, origin_y, px_size, px_size)

    arr = np.array(mosaic)  # (rows, cols, 3)
    height, width = arr.shape[0], arr.shape[1]

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "height": height,
        "width": width,
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "RGB",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for band in range(3):
            dst.write(arr[:, :, band], band + 1)

    print(f"Saved {out_path} ({width}x{height} px)")


def _find_bundled_gdal_dll():
    site_packages = os.path.dirname(os.path.dirname(rasterio.__file__))
    matches = glob.glob(os.path.join(site_packages, "rasterio.libs", "gdal-*.dll"))
    if not matches:
        raise RuntimeError("Could not locate the GDAL DLL bundled with rasterio.")
    return matches[0]


def build_external_pyramid(tif_path, factors=OVERVIEW_FACTORS):
    """Build an external (.ovr) pyramid so GIS apps can render it quickly.

    GDAL only writes overviews to a sidecar .ovr file (instead of embedding
    them in the GeoTIFF) when the dataset is opened read-only. rasterio's
    build_overviews() is only exposed on writable datasets, so we drop down
    to ctypes and call the bundled GDAL C API directly.
    """
    gdal = ctypes.CDLL(_find_bundled_gdal_dll())

    gdal.GDALAllRegister()
    gdal.GDALOpen.restype = ctypes.c_void_p
    gdal.GDALOpen.argtypes = [ctypes.c_char_p, ctypes.c_int]
    gdal.GDALBuildOverviews.restype = ctypes.c_int
    gdal.GDALBuildOverviews.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    gdal.GDALClose.argtypes = [ctypes.c_void_p]

    GA_READONLY = 0
    abs_path = os.path.abspath(tif_path).encode("utf-8")

    dataset = gdal.GDALOpen(abs_path, GA_READONLY)
    if not dataset:
        raise RuntimeError(f"GDAL could not open {tif_path} to build overviews.")

    factor_arr = (ctypes.c_int * len(factors))(*factors)
    err = gdal.GDALBuildOverviews(
        dataset, b"AVERAGE", len(factors), factor_arr, 0, None, None, None
    )
    gdal.GDALClose(dataset)

    if err != 0:
        raise RuntimeError(f"GDALBuildOverviews failed with CPLErr code {err}.")

    ovr_path = tif_path + ".ovr"
    if os.path.exists(ovr_path):
        print(f"Built external pyramid: {ovr_path}")
    else:
        print("Warning: expected external .ovr file was not created.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    x_min, x_max, y_min, y_max = bounding_tiles(AOI_POLYGON, ZOOM)
    print(f"Zoom {ZOOM}: downloading tiles x[{x_min},{x_max}) y[{y_min},{y_max}) "
          f"({(x_max - x_min) * (y_max - y_min)} tiles)")

    mosaic = download_mosaic(x_min, x_max, y_min, y_max, ZOOM)
    save_geotiff(mosaic, x_min, y_min, ZOOM, OUTPUT_TIF)
    build_external_pyramid(OUTPUT_TIF)


if __name__ == "__main__":
    main()
