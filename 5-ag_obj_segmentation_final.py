import ee
ee.Initialize(project="ee-joshisur231")

from helpers import config
from helpers import utils

def main():
    target_crs = "EPSG:32645"
    target_scale = 30

    dem = ee.Image("USGS/SRTMGL1_003").rename("z")
    terrain = ee.Terrain.products(dem)
    seg_bands = ee.List(["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "ndmi", "ndvi_shade", "ndvi_savg"])
    indices_formula = {
        'NDVI': 'clamp((b("nir") - b("red")) / (b("nir") + b("red")), -1, 1)',
        'EVI': (
            'clamp(2.5 * ((b("nir") - b("red")) / '
            '     clamp(b("nir") + 6 * b("red") - 7.5 * b("blue") + 1, 0, 10)), '
            '   -1.0, 1.0)'
        ),
        'NBR': 'clamp((b("nir") - b("swir2")) / (b("nir") + b("swir2")), -1, 1)',
        'NDWI': 'clamp((b("green") - b("nir")) / (b("green") + b("nir")), -1, 1)',
        'NDMI': 'clamp((b("nir") - b("swir1")) / (b("nir") + b("swir1")), -1, 1)',
        'BAI': '1.0 / (pow(0.1 - b("red"), 2) + pow(0.06 - b("nir"), 2))',
        'MSAVI': '(2 * b("nir") + 1 - sqrt(pow(2 * b("nir") + 1, 2) - 8 * (b("nir") - b("red")))) / 2.0',
    }
    
    def add_indices(image):
        ndvi = image.expression(indices_formula["NDVI"]).rename("ndvi")
        ndmi = image.expression(indices_formula["NDMI"]).rename("ndmi")
        evi = image.expression(indices_formula["EVI"]).rename("evi")
        msavi = image.expression(indices_formula["MSAVI"]).rename("msavi")
        return image.addBands(ndvi).addBands(ndmi).addBands(evi).addBands(msavi)

    print("Building seg_image_stack...")
    l_winter2002_image = utils.get_processed_landsat_collection("LANDSAT/LE07/C02/T1_L2", config.ROI, ["2002-12-01", "2003-04-01"], utils.mask_clouds_landsat75, utils.apply_scale_factors).median().select(config.L75_ORIGINAL_BAND_NAMES).rename(config.L75_NEW_BAND_NAMES)
    l_winter2002_image = add_indices(l_winter2002_image)
    high_val_winter2002 = ee.Number(utils.calc_image_stats(l_winter2002_image, config.ROI, config.SCALE).get("high"))
    l_winter2002_image = utils.add_scaled_glcm(l_winter2002_image, config.ROI, config.SCALE, high_val_winter2002).select(seg_bands).rename(seg_bands.map(lambda band_name: ee.String(band_name).cat("_2002")))

    terrain = utils.prepare_terrain_seg(terrain, config.ROI, scale = config.SCALE, high_val=high_val_winter2002)

    l_winter2010_image = utils.get_processed_landsat_collection("LANDSAT/LT05/C02/T1_L2", config.ROI, ["2010-12-01", "2011-03-01"],utils.mask_clouds_landsat75, utils.apply_scale_factors).median().select(config.L75_ORIGINAL_BAND_NAMES).rename(config.L75_NEW_BAND_NAMES)
    l_winter2010_image = add_indices(l_winter2010_image)
    high_val_winter2010 = ee.Number(utils.calc_image_stats(l_winter2010_image, config.ROI, config.SCALE).get("high"))
    l_winter2010_image = utils.add_scaled_glcm(l_winter2010_image, config.ROI, config.SCALE, high_val_winter2010).select(seg_bands).rename(seg_bands.map(lambda band_name: ee.String(band_name).cat("_2010")))

    l_preMons_image = utils.get_processed_landsat_collection("LANDSAT/LC08/C02/T1_L2", config.ROI, ["2017-03-01", "2017-06-01"], utils.mask_clouds_landsat8, utils.apply_scale_factors).median().select(config.L8_ORIGINAL_BAND_NAMES).rename(config.L8_NEW_BAND_NAMES)
    l_preMons_image = add_indices(l_preMons_image)
    high_val_preMons = ee.Number(utils.calc_image_stats(l_preMons_image, config.ROI, config.SCALE).get("high"))
    l_preMons_image = utils.add_scaled_glcm(l_preMons_image, config.ROI, config.SCALE, high_val_preMons).select(seg_bands).rename(seg_bands.map(lambda band_name: ee.String(band_name).cat("_2017")))

    seg_image_stack = l_winter2002_image.addBands(l_winter2010_image).addBands(l_preMons_image).addBands(terrain)

    common_params = {"s": 4, "n": 8, "cn": 8, "cm": 0.1}
    out_dir = "projects/ee-joshisur231/assets/agriculture_abandonment_nepal/image_seg_results_single"

    region_stack = seg_image_stack 
    
    seeds = ee.Algorithms.Image.Segmentation.seedGrid(common_params["s"], "hex").reproject(crs=target_crs, scale=target_scale)
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=region_stack.reproject(crs=target_crs, scale=target_scale),
        connectivity=common_params["cn"],
        compactness=common_params["cm"],
        neighborhoodSize=common_params["n"],
        seeds=seeds
    )
    
    clusters = snic.select('clusters').rename('clusters').int32()

    asset_id = f"{out_dir}/aal_objs"
    desc = f"aal_objs"
    
    task = ee.batch.Export.image.toAsset(
        image=clusters,
        description=desc,
        assetId=asset_id,
        region=config.ROI,
        scale=target_scale,
        crs=target_crs,
        maxPixels=1e13
    )
    task.start()
    print(f"Submitted task: {desc} -> {asset_id}")

    print("All tasks submitted successfully!")

if __name__ == "__main__":
    main()
