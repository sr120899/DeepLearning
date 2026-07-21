"""
YOLO-Extraction.py

Extract building footprints from GoogleMap-Images.tif using a YOLOv8
instance-segmentation model fine-tuned for buildings, and save:

    YOLO-Extraction.tif       - binary building mask, georeferenced
    YOLO-Extraction.geojson   - vectorized building footprint polygons

The model is tiled across the input raster in TILE_SIZE x TILE_SIZE chunks
(matching the model's training resolution) since the input image is
typically larger than a single YOLO input tile.

Usage:
    python YOLO-Extraction.py

Requirements (already installed in .venv):
    ultralytics, huggingface_hub, rasterio, shapely

Note: on first run this downloads the "keremberke/yolov8n-building-
segmentation" checkpoint (~6MB) from the Hugging Face Hub.
"""

import json
import os

import numpy as np
import rasterio
import shapely.geometry
from huggingface_hub import hf_hub_download
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from ultralytics import YOLO

INPUT_TIF = "GoogleMap-Images.tif"
OUTPUT_TIF = "YOLO-Extraction.tif"
OUTPUT_GEOJSON = "YOLO-Extraction.geojson"

MODEL_REPO = "keremberke/yolov8n-building-segmentation"
MODEL_FILE = "best.pt"
TILE_SIZE = 640          # matches the model's training input size
CONF_THRESHOLD = 0.25
SIMPLIFY_TOLERANCE = None  # e.g. 0.00001 (degrees) to reduce polygon vertices


def load_model():
    weights_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    return YOLO(weights_path)


def iter_tiles(width, height, tile_size):
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            yield x, y, w, h


def detect_buildings(model, image):
    """Run tiled YOLO segmentation over `image` (H, W, 3 RGB) and return a
    binary (H, W) uint8 mask (255 = building)."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)

    for x, y, w, h in iter_tiles(width, height, TILE_SIZE):
        tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        tile[:h, :w] = image[y : y + h, x : x + w]
        tile_bgr = tile[:, :, ::-1]  # ultralytics expects BGR for raw arrays

        result = model.predict(tile_bgr, conf=CONF_THRESHOLD, imgsz=TILE_SIZE, verbose=False)[0]
        if result.masks is None:
            continue

        for tile_mask in result.masks.data.cpu().numpy():
            mask[y : y + h, x : x + w] |= tile_mask[:h, :w] > 0.5

    return (mask * 255).astype(np.uint8)


def save_mask_geotiff(mask, transform, crs, out_path):
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": mask.shape[0],
        "width": mask.shape[1],
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)


def vectorize_mask(mask_path, out_path, simplify_tolerance=None):
    """Vectorize a binary raster mask to a WGS84 GeoJSON FeatureCollection.

    Written directly with rasterio + shapely/json rather than
    geopandas.to_file(), since geopandas' pyogrio/fiona I/O backends can't
    locate a usable GDAL install in this environment; rasterio's own
    bundled GDAL (already used elsewhere in this pipeline) does the shape
    extraction and reprojection instead.
    """
    with rasterio.open(mask_path) as src:
        band = src.read(1)
        mask = band != 0
        results = list(rio_shapes(band, mask=mask, transform=src.transform))
        crs = src.crs

    features = []
    for geom, value in results:
        if simplify_tolerance is not None:
            geom = shapely.geometry.mapping(
                shapely.geometry.shape(geom).simplify(simplify_tolerance)
            )
        if crs is not None and crs.to_epsg() != 4326:
            geom = transform_geom(crs, "EPSG:4326", geom)
        features.append(
            {"type": "Feature", "geometry": geom, "properties": {"value": value}}
        )

    feature_collection = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f)


def main():
    if not os.path.exists(INPUT_TIF):
        raise FileNotFoundError(
            f"{INPUT_TIF} not found - run GoogleMap-Download.py first."
        )

    with rasterio.open(INPUT_TIF) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs

    print(f"Loading YOLO model ({MODEL_REPO})...")
    model = load_model()

    print(f"Running tiled building detection ({TILE_SIZE}x{TILE_SIZE} tiles)...")
    mask = detect_buildings(model, image)
    n_building_px = int((mask > 0).sum())
    print(f"Building pixels detected: {n_building_px} ({n_building_px / mask.size:.1%} of image)")

    save_mask_geotiff(mask, transform, crs, OUTPUT_TIF)
    print(f"Saved mask: {OUTPUT_TIF}")

    vectorize_mask(OUTPUT_TIF, OUTPUT_GEOJSON, simplify_tolerance=SIMPLIFY_TOLERANCE)
    print(f"Saved vector: {OUTPUT_GEOJSON}")


if __name__ == "__main__":
    main()
