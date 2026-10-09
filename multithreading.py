"""
Parallel local download of per-tile predictor GeoTIFFs from Earth Engine.

Same pattern as the GEDI extraction script: high-volume endpoint + a thread pool
(the work is I/O-bound waiting on EE), plus skip-if-exists, retries with backoff,
and a failure log so the script can be safely re-run.

Usage:
    python download_landsat_tiles.py --test            # one tile, prints timing + raster info
    python download_landsat_tiles.py                   # everything
    python download_landsat_tiles.py --workers 6 --years 2012 2013
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from gee_init import initialize_gee
initialize_gee(account="work1", project="ee-municipalites-climate")
ee.Initialize(project="ee-municipalites-climate", opt_url="https://earthengine-highvolume.googleapis.com")

import geemap
from helpers import landsat_composites

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SCALE = 30
CRS = "EPSG:32645"
YEARS = [2008, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
ROOT = "projects/ee-joshisur231/assets/agriculture_abandonment_nepal_revised"
TILE_M = 90_000

OUT_DIR = r"E:\work\agriculture_abandonment\landsat_tiles"   # <-- edit
FAIL_LOG = os.path.join(OUT_DIR, "failed_tiles.txt")

MAX_WORKERS = 4        # outer threads; each download is ALSO concurrent internally (geedim)
MAX_TILE_DIM = 512     # px; caps the size/compute of each underlying EE request
MAX_RETRIES = 4
# Extra kwargs for geemap.download_ee_image. Check that max_requests is actually
# forwarded to geedim before using it:
#   import inspect; print(inspect.getsource(geemap.download_ee_image))
DOWNLOAD_KWARGS = {}   # e.g. {"max_requests": 8}

# ----------------------------------------------------------------------------
# Image definition (unchanged from the export pipeline)
# ----------------------------------------------------------------------------
ROI = ee.FeatureCollection("projects/ee-joshisur231/assets/pa_effectiveness/nepal_boundary").first().geometry()
slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).rename("slope")
tri_dtm = ee.Image(f"{ROOT}/terrain_roughness_DTM").rename("tri_dtm")

processor = landsat_composites.C2SRExpressions()


def add_tile_uid(tiles_fc):
    """Adds a sequential integer 'tile_id' to a FeatureCollection."""
    tile_count = tiles_fc.size()
    tiles_list = tiles_fc.toList(tile_count)
    fc_with_id = ee.List.sequence(0, tile_count.subtract(1)).map(
        lambda i: ee.Feature(tiles_list.get(i)).set("tile_id", ee.Number(i).int())
    )
    return ee.FeatureCollection(fc_with_id)


def compute_indices(image):
    ndvi = image.expression(processor.ALGORITHMS["NDVI"]).rename("ndvi")
    evi = image.expression(processor.ALGORITHMS["EVI"]).rename("evi")
    ndmi = image.expression(processor.ALGORITHMS["NDMI"]).rename("ndmi")
    msavi = image.expression(processor.ALGORITHMS["MSAVI"]).rename("msavi")
    return image.addBands([ndvi, evi, ndmi, msavi])


unified_collection = (
    processor._expression()
    .map(compute_indices)
    .select(["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "evi", "ndmi", "msavi"])
)

combined_reducer = ee.Reducer.median().combine(ee.Reducer.percentile([25, 75]), sharedInputs=True)


def get_3yr_predictors(target_year):
    target_year = ee.Number(target_year)
    start_date = ee.Date.fromYMD(target_year.subtract(1), 1, 1)
    end_date = ee.Date.fromYMD(target_year.add(2), 1, 1)

    window_col = unified_collection.filterDate(start_date, end_date)
    pixel_count = window_col.select("ndvi").reduce(ee.Reducer.count()).rename("px_count")
    quality_flag = (
        ee.Image(0)
        .where(pixel_count.gte(6), 1)
        .where(pixel_count.gte(12), 2)
    ).rename("quality_flag")
    predictors = window_col.reduce(combined_reducer)
    ndvi_iqr = predictors.select("ndvi_p75").subtract(predictors.select("ndvi_p25")).rename("ndvi_iqr")
    evi_iqr = predictors.select("evi_p75").subtract(predictors.select("evi_p25")).rename("evi_iqr")
    msavi_iqr = predictors.select("msavi_p75").subtract(predictors.select("msavi_p25")).rename("msavi_iqr")

    return (
        predictors.addBands(ndvi_iqr)
        .addBands(evi_iqr)
        .addBands(msavi_iqr)
        .addBands(slope)
        .addBands(tri_dtm)
        .addBands(pixel_count)
        .addBands(quality_flag)
        .set("year", target_year)
        .set("system:time_start", ee.Date.fromYMD(target_year, 1, 1).millis())
    )


# ----------------------------------------------------------------------------
# Parallel download
# ----------------------------------------------------------------------------
_log_lock = threading.Lock()


def log_failure(desc, err):
    with _log_lock:
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            f.write(f"{desc}: {err}\n")


def tile_path(year, tile_id):
    return os.path.join(OUT_DIR, f"pred_{year}_{tile_id}.tif")


def download_one(year, tile_id, region, image):
    """Download one (year, tile). Writes to a .partial file and renames on success,
    so an interrupted run never leaves a truncated .tif that would be skipped later."""
    desc = f"pred_{year}_{tile_id}"
    final = tile_path(year, tile_id)
    if os.path.exists(final):
        return desc, "skipped", 0.0

    tmp = final.replace(".tif", ".partial.tif")
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            geemap.download_ee_image(
                image,
                tmp,
                region=region,
                crs=CRS,
                scale=SCALE,
                dtype="float32",
                overwrite=True,
                max_tile_dim=MAX_TILE_DIM,
                **DOWNLOAD_KWARGS,
            )
            os.replace(tmp, final)
            return desc, "ok", time.time() - t0
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                log_failure(desc, repr(e))
                return desc, f"failed ({type(e).__name__})", time.time() - t0
            time.sleep(min(120, 5 * 2 ** attempt))  # 10s, 20s, 40s ...


def describe_raster(path):
    """Pilot-run sanity check: confirm shape / bounds / pixel size look right."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            print(f"  shape={src.height}x{src.width}, bands={src.count}, dtype={src.dtypes[0]}")
            print(f"  bounds={tuple(round(b, 1) for b in src.bounds)}, res={src.res}, crs={src.crs}")
    except ImportError:
        print("  (install rasterio to print raster info)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="download a single tile and print timing")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--years", type=int, nargs="*", default=YEARS)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    tiles = add_tile_uid(ROI.coveringGrid(ee.Projection(CRS), TILE_M))
    tile_features = tiles.getInfo()["features"]
    print(f"{len(tile_features)} tiles x {len(args.years)} years")

    # Build each year's image once, on the main thread, and share it across workers.
    images = {y: get_3yr_predictors(y) for y in args.years}

    jobs = []
    for year in args.years:
        for f in tile_features:
            tile_id = f["properties"]["tile_id"]
            region = ee.Geometry(f["geometry"])
            jobs.append((year, tile_id, region, images[year]))

    if args.test:
        jobs = jobs[:1]

    t_start = time.time()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, *job): job for job in jobs}
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                desc, status, secs = fut.result()
                key = "failed" if status.startswith("failed") else status
                counts[key] += 1
                print(f"[{i}/{len(jobs)}] {desc}: {status} ({secs:.0f}s)")
                if args.test and status == "ok":
                    describe_raster(tile_path(futures[fut][0], futures[fut][1]))
        except KeyboardInterrupt:
            print("Interrupted, cancelling queued downloads...")
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    print(f"Done in {time.time() - t_start:.0f}s: {counts}")
    if counts["failed"]:
        print(f"See {FAIL_LOG} for failures; re-run the script to retry only those.")


if __name__ == "__main__":
    main()