##Resample domain outputs by percentile
#version2

import arcpy
from arcpy.sa import *
import numpy as np
import os

arcpy.CheckOutExtension("Spatial")

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

PROJECT_DIR = r"Project Directory"

# prediction run holding the prob_XXka rasters to rescale
PRED_GDB = os.path.join(PROJECT_DIR, "run_XXX", "predictions.gdb")

COAST_DIR = os.path.join(PROJECT_DIR, "output_uncorrected_run3")
TERRAIN_GDB = os.path.join(PROJECT_DIR, "run_XXX", "step4c_twi_slope.gdb")
HABITABILITY_GDB = os.path.join(PROJECT_DIR, "run_XXX", "step5_habitability_bool.gdb")

DOMAIN_KA = 30
TIMESTEPS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# near-modern coastline defines the survey boundary, not each timestep's own coast -
# the split that matters is whether an area has been available for terrestrial survey
BOUNDARY_KA = 1

# step5 classes dropped before ranking, so the percentile scale describes land only
#lake interiors score near zero and would otherwise fill the bottom of the distribution
# 0 habitable, 1 ice, 2 inundated, 3 lake, 4 melt channel, 5 fluvial
RESCALE_EXCLUDE_CLASSES = [3, 4]

OUTPUT_SCALE = 1.0     # set to 100 for percentile points rather than 0-1


def get_run_folder():
    existing = [d for d in os.listdir(PROJECT_DIR) if d.startswith("run_") and d[4:].isdigit()]
    next_num = max([int(d[4:]) for d in existing], default=0) + 1
    run_folder = os.path.join(PROJECT_DIR, f"run_{next_num:03d}")
    os.makedirs(run_folder)
    arcpy.management.CreateFileGDB(run_folder, "prob_rescaled.gdb")
    return os.path.join(run_folder, "prob_rescaled.gdb")


def pin_environment():
    grid = os.path.join(TERRAIN_GDB, f"twi_{DOMAIN_KA}ka")
    arcpy.env.snapRaster = grid
    arcpy.env.cellSize = grid
    arcpy.env.extent = arcpy.Describe(grid).extent
    arcpy.env.outputCoordinateSystem = arcpy.Describe(grid).spatialReference
    arcpy.env.resamplingMethod = "NEAREST"
    ref = arcpy.Raster(grid)
    return ref.extent.lowerLeft, ref.meanCellWidth, (ref.height, ref.width)


def to_arr(path, target):
    # Float() forces map algebra so the read lands on the env grid rather than the source grid
    arr = arcpy.RasterToNumPyArray(Float(arcpy.Raster(path)),
                                   nodata_to_value=-9999).astype(np.float32)
    if arr.shape != target:
        raise RuntimeError(f"{path} returned {arr.shape}, expected {target}")
    arr[arr == -9999] = np.nan
    return arr


# fraction of cells in the same domain scoring at or below each value, ties share a rank
def percentile_rank(vals):
    srt = np.sort(vals)
    return np.searchsorted(srt, vals, side="right") / len(srt) * OUTPUT_SCALE


def process_timestep(ka, coast, out_gdb, ll, cell, shape):
    prob = to_arr(os.path.join(PRED_GDB, f"prob_{ka}ka"), shape)

    valid = ~np.isnan(prob)
    if RESCALE_EXCLUDE_CLASSES:
        hab = to_arr(os.path.join(HABITABILITY_GDB, f"habitability_bool_{ka}ka"), shape)
        valid &= ~np.isin(hab, RESCALE_EXCLUDE_CLASSES)

    terr = valid & (coast == 0)
    marine = valid & (coast == 1)

    out_combined = np.full(shape, np.nan, dtype=np.float32)
    out_terr = np.full(shape, np.nan, dtype=np.float32)
    out_marine = np.full(shape, np.nan, dtype=np.float32)

    for mask, single in [(terr, out_terr), (marine, out_marine)]:
        n = int(mask.sum())
        if n == 0:
            continue
        pct = percentile_rank(prob[mask])
        single[mask] = pct
        out_combined[mask] = pct

    combined_ras = arcpy.NumPyArrayToRaster(out_combined, ll, cell, cell, value_to_nodata=np.nan)
    combined_ras.save(os.path.join(out_gdb, f"pctl_combined_{ka}ka"))

    terr_ras = arcpy.NumPyArrayToRaster(out_terr, ll, cell, cell, value_to_nodata=np.nan)
    terr_ras.save(os.path.join(out_gdb, f"pctl_terrestrial_{ka}ka"))

    marine_ras = arcpy.NumPyArrayToRaster(out_marine, ll, cell, cell, value_to_nodata=np.nan)
    marine_ras.save(os.path.join(out_gdb, f"pctl_marine_{ka}ka"))

    print(f"  {ka}ka: terrestrial {int(terr.sum()):>8}  marine {int(marine.sum()):>8}")


def main():
    out_gdb = get_run_folder()
    print(f"writing to {out_gdb}")

    ll, cell, shape = pin_environment()

    # boundary is time-invariant so it loads once
    coast_raw = to_arr(os.path.join(COAST_DIR, f"Coast_{BOUNDARY_KA}ka.tif"), shape)
    coast = np.where(np.isnan(coast_raw), 0, coast_raw)

    for ka in TIMESTEPS:
        process_timestep(ka, coast, out_gdb, ll, cell, shape)

    print(f"\nall done, {len(TIMESTEPS) * 3} rasters in {out_gdb}")


if __name__ == "__main__":
    main()
