"""
GeoSAM-Extraction.py

Extract building footprints from GoogleMap-Images.tif using a text-prompted
Segment Anything Model (LangSAM = GroundingDINO for "building" detection +
SAM for pixel-accurate masks), and save:

    GeoSAM-Extraction.tif       - binary building mask, georeferenced
    GeoSAM-Extraction.geojson   - vectorized building footprint polygons

Detection is tiled across the raster in TILE_SIZE x TILE_SIZE chunks rather
than run once on the full image. GroundingDINO internally downsizes its
input to an ~800px short side; feeding it the whole (larger) scene at once
shrinks small rooftops below its detection floor, so tiling first preserves
enough per-building pixel detail to catch them.

Usage:
    python GeoSAM-Extraction.py

Requirements (already installed in .venv):
    segment-geospatial, groundingdino-py, torch, torchvision, rasterio,
    geopandas, shapely

Note: on first run this downloads the GroundingDINO (SwinB, ~938MB) and
SAM (vit_b, ~375MB) checkpoints to the Hugging Face / torch cache. There
is no GPU on this machine, so inference runs on CPU and may take a while.
"""

import json
import os

import numpy as np
import rasterio
import shapely.geometry
import torch
from PIL import Image
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from transformers.modeling_utils import ModuleUtilsMixin

# groundingdino-py (last released for transformers ~4.x) calls two
# ModuleUtilsMixin APIs the way they looked back then. Newer transformers
# releases changed both, so we shim them here before GroundingDINO builds
# its BERT text encoder.

# 1) get_head_mask() was removed entirely.
if not hasattr(ModuleUtilsMixin, "get_head_mask"):

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return head_mask.to(dtype=self.dtype)

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    ModuleUtilsMixin._convert_head_mask_to_5d = _convert_head_mask_to_5d
    ModuleUtilsMixin.get_head_mask = _get_head_mask

# 2) get_extended_attention_mask()'s 3rd positional argument used to be
# `device`; it is now `dtype`. groundingdino-py still calls it positionally
# as (attention_mask, input_shape, device), so re-wrap it to accept either
# calling convention.
_original_get_extended_attention_mask = ModuleUtilsMixin.get_extended_attention_mask


def _compat_get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    if isinstance(device, torch.dtype):
        dtype = device
        device = None
    return _original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)


ModuleUtilsMixin.get_extended_attention_mask = _compat_get_extended_attention_mask

from samgeo.text_sam import LangSAM

INPUT_TIF = "GoogleMap-Images.tif"
OUTPUT_TIF = "GeoSAM-Extraction.tif"
OUTPUT_GEOJSON = "GeoSAM-Extraction.geojson"

TEXT_PROMPT = "building . rooftop . roof"  # multiple synonyms improve recall
BOX_THRESHOLD = 0.24     # min confidence for a GroundingDINO detection box
TEXT_THRESHOLD = 0.24    # min confidence for the text/box match
SAM_MODEL_TYPE = "vit_b"  # smallest/fastest SAM 1 checkpoint (CPU-friendly)
SIMPLIFY_TOLERANCE = None  # e.g. 0.00001 (degrees) to reduce polygon vertices
MAX_BOX_AREA_FRACTION = 0.5  # reject GroundingDINO boxes covering more of a
# tile than this - at low thresholds it occasionally emits one degenerate
# box spanning nearly the whole tile, which SAM then turns into a mask that
# blankets it.
TILE_SIZE = 640  # run detection per-tile at this resolution (see module docstring)


def raster_to_geojson(mask_tif_path, output_path, simplify_tolerance=None):
    """Vectorize a binary raster mask to a WGS84 GeoJSON FeatureCollection.

    Bypasses geopandas' to_file() (pyogrio/fiona), which fails to locate a
    usable GDAL install in this environment; rasterio's own bundled GDAL
    (already relied on elsewhere in this pipeline) does the shape
    extraction and reprojection instead.
    """
    with rasterio.open(mask_tif_path) as src:
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f)


def iter_tiles(width, height, tile_size):
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            yield x, y, w, h


def reject_oversized_boxes(tile_area):
    def _filter(box, mask, logit, phrase, index):
        x0, y0, x1, y1 = (float(v) for v in box)
        box_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        return box_area <= tile_area * MAX_BOX_AREA_FRACTION

    return _filter


def detect_buildings(model, image):
    """Run tiled LangSAM detection over `image` (H, W, 3 RGB) and return a
    binary (H, W) uint8 mask (255 = building)."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    tile_area = TILE_SIZE * TILE_SIZE

    for x, y, w, h in iter_tiles(width, height, TILE_SIZE):
        tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        tile[:h, :w] = image[y : y + h, x : x + w]
        tile_pil = Image.fromarray(tile)

        model.predict(
            tile_pil,
            TEXT_PROMPT,
            BOX_THRESHOLD,
            TEXT_THRESHOLD,
            detection_filter=reject_oversized_boxes(tile_area),
        )
        tile_mask = model.prediction
        if tile_mask is not None:
            mask[y : y + h, x : x + w] |= tile_mask[:h, :w] > 0

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


def main():
    if not os.path.exists(INPUT_TIF):
        raise FileNotFoundError(
            f"{INPUT_TIF} not found - run GoogleMap-Download.py first."
        )

    with rasterio.open(INPUT_TIF) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform = src.transform
        crs = src.crs

    print(f"Loading LangSAM (GroundingDINO + SAM {SAM_MODEL_TYPE})...")
    model = LangSAM(model_type=SAM_MODEL_TYPE)

    print(f"Running tiled detection ({TILE_SIZE}x{TILE_SIZE}) with text prompt: '{TEXT_PROMPT}'...")
    mask = detect_buildings(model, image)
    n_building_px = int((mask > 0).sum())
    print(f"Building pixels detected: {n_building_px} ({n_building_px / mask.size:.1%} of image)")

    save_mask_geotiff(mask, transform, crs, OUTPUT_TIF)
    print(f"Saved mask: {OUTPUT_TIF}")

    raster_to_geojson(OUTPUT_TIF, OUTPUT_GEOJSON, simplify_tolerance=SIMPLIFY_TOLERANCE)
    print(f"Saved vector: {OUTPUT_GEOJSON}")


if __name__ == "__main__":
    main()
