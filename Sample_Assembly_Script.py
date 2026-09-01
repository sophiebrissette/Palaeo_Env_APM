##sample assembly script
#final version
# most recent run - using distance to water, hoebe sites and geology

import arcpy
from arcpy.sa import *
import os
import csv

arcpy.CheckOutExtension("Spatial")

arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

# paths to the per-timestep predictor stacks, all in bng at 500m
PROJECT_DIR = r"Define Path"

# set these to the actual run numbers 
LAKE_DEPTH_GDB = os.path.join(PROJECT_DIR, "run_XXX", "lake_depth.gdb")
GEOLOGY_GDB = os.path.join(PROJECT_DIR, "Outputs", "run_XXX", "geology.gdb")
DIST_GDB = os.path.join(PROJECT_DIR, "run_XXX", "dist_water.gdb")

COAST_DIR = os.path.join(PROJECT_DIR, "XXX")
ICE_MASK_GDB = os.path.join(PROJECT_DIR, "XXX", "ice_flag.gdb")
TERRAIN_GDB = os.path.join(PROJECT_DIR, "XXX", "step4c_twi_slope.gdb")
HYDRO_GDB = os.path.join(PROJECT_DIR, "XXX", "step4b_meltwater_lakes.gdb")
LAKE_GDB = os.path.join(PROJECT_DIR, "XXX", "lake_binary.gdb")
ICE_GDB = os.path.join(PROJECT_DIR, "XXX", "ice_layers.gdb")
SITE_FC = os.path.join(PROJECT_DIR, "XXX", "Cleaned_Site_Data")

# any layer at 500m bng works as the env anchor, twi_30ka is arbitrary
REF_RASTER = os.path.join(TERRAIN_GDB, "twi_30ka")

#these are the timesteps modeled in the project - they can be altered individually or, given the appropriate environmental
# and palaeotopohraphic reconstructions are modelled, any period
#future work could include modelling different timestep intervals such as Hoebe et al. 2024
TIMESTEPS = [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]

# british-irish ice sheet extant across these ka only, ice predictors zero-fill outside
ICE_TIMESTEPS = set(range(15, 32))

BG_RATIO = 1.0          # pseudo-absences per valid presence
BG_BUFFER_M = 500      # exclusion buffer around presences
BG_MIN_SPACING_M = 500  # minimum distance between pseudo-absence points

DOMAIN_KA = 30          # fix the spatial domain to this timestep's twi footprint, applied to all ka

# target-group background - restrict pseudo-absences to the same surveyed surface the
# presences come from, so survey footprint appears in both classes and largely cancels
# (phillips et al. 2009). set False to sample background from the whole habitable domain
TARGET_GROUP_BACKGROUND = True
SURVEY_MASK_KA = 1      # coast tif defines the surveyed land surface

# status codes written to the csv
# 1 = valid presence
# 0 = pseudo-absence
# 99 = presence located on ice or under water at this timestep, dates need review


# auto-numbered run folder with a fresh gdb, matches the max+1 convention across project
def get_run_folder():
    existing = [d for d in os.listdir(PROJECT_DIR) if d.startswith("run_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_num = max(nums) + 1 if nums else 1
    run_folder = os.path.join(PROJECT_DIR, f"run_{next_num:03d}")
    os.makedirs(run_folder)
    arcpy.management.CreateFileGDB(run_folder, "samples.gdb")
    return run_folder, os.path.join(run_folder, "samples.gdb")


# pin arcpy env to a specific raster so downstream ops share snap, cell size, extent
# crs called once per timestep to guard against session-state carrying over
def reset_environment(reference_raster):
    arcpy.env.snapRaster = reference_raster
    arcpy.env.cellSize = reference_raster
    arcpy.env.extent = reference_raster
    arcpy.env.outputCoordinateSystem = arcpy.Describe(reference_raster).spatialReference


# one-time reprojection of the site fc from wgs84 lat/lon to bng
def project_sites(out_gdb):
    ref_sr = arcpy.Describe(REF_RASTER).spatialReference
    projected = os.path.join(out_gdb, "sites_bng")
    arcpy.management.Project(SITE_FC, projected, ref_sr)
    return projected


# habitable-land mask for this ka: dry (coast=0) and, if ice existed, not under ice
# masked to the twi footprint so no habitat cell exists where the terrain stack has no data
# for_background=True additionally restricts to the surveyed land surface, so pseudo-absences
# are drawn from the same biased surface as the presences rather than the whole palaeo-domain
def build_habitat(ka, for_background=False):
    # domain fixed 
    domain_twi = Raster(os.path.join(TERRAIN_GDB, f"twi_{DOMAIN_KA}ka"))
    coast_raw = Raster(os.path.join(COAST_DIR, f"Coast_{ka}ka.tif"))
    coast = Con(IsNull(coast_raw), 0, coast_raw)   # nodata outside coast footprint = habitable
    cond = (~IsNull(domain_twi)) & (coast == 0)   # single shared spatial domain across all ka

    if ka in ICE_TIMESTEPS:
        ice_raw = Raster(os.path.join(ICE_MASK_GDB, f"Binary_Ice_{ka}ka"))
        ice = Con(IsNull(ice_raw), 0, ice_raw)   # nodata outside ice footprint = no ice
        cond = cond & (ice == 0)

    if for_background and TARGET_GROUP_BACKGROUND:
        cond = cond & (build_survey_mask() == 1)

    return Con(cond, 1)


# surveyed surface = land at the near-modern coastline, which is where every recorded site
# in both the ads and radiocarbon datasets was found
def build_survey_mask():
    coast_raw = Raster(os.path.join(COAST_DIR, f"Coast_{SURVEY_MASK_KA}ka.tif"))
    coast = Con(IsNull(coast_raw), 1, coast_raw)   # nodata treated as sea, keeps the mask tight
    return Con(coast == 0, 1, 0)


# attach one column per predictor raster to the input point fc
# on_ice/ice_thick/ice_vel only appear for glacial timesteps, collect_rows zero-fills them
def extract_predictors(pts, ka):
    layers = [
        [os.path.join(TERRAIN_GDB, f"twi_{ka}ka"), "twi"],
        [os.path.join(TERRAIN_GDB, f"slope_{ka}ka"), "slope"],
        [os.path.join(HYDRO_GDB, f"flow_acc_fluvial_{ka}ka"), "faf"],
        [os.path.join(HYDRO_GDB, f"head_surface_{ka}ka"), "head_surf"],
        [os.path.join(LAKE_GDB, f"lake_bin_{ka}ka"), "lake_bin"],
        [os.path.join(LAKE_DEPTH_GDB, f"lake_depth_{ka}ka"), "lake_depth"],
        [os.path.join(COAST_DIR, f"Coast_{ka}ka.tif"), "on_water"],
        [os.path.join(GEOLOGY_GDB, "geology_litho"), "geology"],
        [os.path.join(DIST_GDB, f"dist_coast_{ka}ka"), "dist_coast"],
        [os.path.join(DIST_GDB, f"dist_lake_{ka}ka"), "dist_lake"],
        [os.path.join(DIST_GDB, f"dist_river_{ka}ka"), "dist_river"],
    ]

    # ice_combined is a 3-band raster, extract thickness (band 1) and velocity (band 3)
    if ka in ICE_TIMESTEPS:
        ice_ras = os.path.join(ICE_GDB, f"ice_combined_{ka}ka")
        layers += [
            [os.path.join(ice_ras, "Band_1"), "ice_thick"],
            [os.path.join(ice_ras, "Band_3"), "ice_vel"],
            [os.path.join(ICE_MASK_GDB, f"Binary_Ice_{ka}ka"), "on_ice"],
        ]

    ExtractMultiValuesToPoints(pts, layers, "NONE")


# write the status code directly onto the fc so it can be symbolised and filtered in the map
# 1 = valid presence, 0 = pseudo-absence, 99 = presence on ice or under water at this ka
def add_status(pts, is_presence):
    if "status" not in {f.name for f in arcpy.ListFields(pts)}:
        arcpy.management.AddField(pts, "status", "SHORT")

    if not is_presence:
        with arcpy.da.UpdateCursor(pts, ["status"]) as cur:
            for row in cur:
                row[0] = 0
                cur.updateRow(row)
        return

    existing = {f.name for f in arcpy.ListFields(pts)}
    has_ice = "on_ice" in existing
    has_water = "on_water" in existing
    fields = ["status"] + (["on_ice"] if has_ice else []) + (["on_water"] if has_water else [])

    with arcpy.da.UpdateCursor(pts, fields) as cur:
        for row in cur:
            on_ice = 0
            on_water = 0
            idx = 1
            if has_ice:
                on_ice = row[idx] or 0
                idx += 1
            if has_water:
                on_water = row[idx] or 0
            row[0] = 99 if (on_ice >= 1 or on_water >= 1) else 1
            cur.updateRow(row)


# filter sites to those whose date range covers this ka, copy to a per-timestep fc, extract
# a site with span [date_until, date_from] in years bp appears in every ka where
# date_until <= ka*1000 <= date_from
def process_presence(sites_bng, ka, out_gdb):
    year_bp = ka * 1000
    where = f"date_until <= {year_bp} AND date_from >= {year_bp}"

    lyr_name = f"sel_{ka}"
    arcpy.management.MakeFeatureLayer(sites_bng, lyr_name, where)
    n = int(arcpy.management.GetCount(lyr_name)[0])
    if n == 0:
        arcpy.management.Delete(lyr_name)
        return None, 0

    pts = os.path.join(out_gdb, f"pres_{ka}ka")
    arcpy.management.CopyFeatures(lyr_name, pts)
    arcpy.management.Delete(lyr_name)

    extract_predictors(pts, ka)
    add_status(pts, is_presence=True)
    return pts, n


# generate pseudo-absence points from the habitable domain, excluding a buffer around presences
# steps: rasterise habitat -> buffer presences -> erase buffer from habitat -> dissolve to
# a single multipart feature (otherwise CreateRandomPoints emits n_bg per polygon part) ->
# random points -> extract predictors
def process_background(habitat_ras, pres_pts, n_bg, ka, out_gdb):
    hab_poly = os.path.join(out_gdb, f"hab_poly_{ka}ka")
    arcpy.conversion.RasterToPolygon(habitat_ras, hab_poly, "NO_SIMPLIFY", "Value")

    if pres_pts:
        buf = os.path.join(out_gdb, f"pres_buf_{ka}ka")
        arcpy.analysis.Buffer(pres_pts, buf, f"{BG_BUFFER_M} Meters", "FULL", "ROUND", "ALL")
        erased = os.path.join(out_gdb, f"hab_erased_{ka}ka")
        arcpy.analysis.Erase(hab_poly, buf, erased)
    else:
        erased = hab_poly

    # dissolve to one multipart feature - otherwise CreateRandomPoints generates n_bg per part
    available = os.path.join(out_gdb, f"hab_avail_{ka}ka")
    arcpy.analysis.PairwiseDissolve(erased, available, "", "", "MULTI_PART")

    bg_name = f"bg_{ka}ka"
    arcpy.management.CreateRandomPoints(out_gdb, bg_name, available, "",
                                        n_bg, f"{BG_MIN_SPACING_M} Meters")
    bg_pts = os.path.join(out_gdb, bg_name)

    extract_predictors(bg_pts, ka)
    add_status(bg_pts, is_presence=False)
    return bg_pts


# read a point fc into a list of dicts ready for csv output
# handles presence vs pseudo-absence differences, zero-fills missing ice predictors,
# and assigns the status code based on on_ice / on_water
def collect_rows(pts, ka, is_presence):
    predictor_fields = ["twi", "slope", "faf", "head_surf", "lake_bin", "lake_depth",
                        "on_water", "geology", "ice_thick", "ice_vel", "on_ice", "status",
                        "dist_coast", "dist_lake", "dist_river"]

    # only pull fields that actually exist on this fc (ice fields absent post-15 ka)
    existing = {f.name for f in arcpy.ListFields(pts)}
    fields = ["SHAPE@XY"]
    if is_presence:
        fields += ["site_name", "date_from", "date_until"]
    fields += [f for f in predictor_fields if f in existing]

    rows = []
    with arcpy.da.SearchCursor(pts, fields) as cur:
        for row in cur:
            d = {"x": row[0][0], "y": row[0][1], "timestep_ka": ka}

            # presence rows carry site identity and dating info, pseudo-absences leave them blank
            if is_presence:
                d["site_name"] = row[1]
                d["date_from"] = row[2]
                d["date_until"] = row[3]
                d["date_span"] = (row[2] - row[3]) if (row[2] is not None and row[3] is not None) else ""
                offset = 4
            else:
                d["site_name"] = ""
                d["date_from"] = ""
                d["date_until"] = ""
                d["date_span"] = ""
                offset = 1

            for i, f in enumerate(fields[offset:]):
                d[f] = row[offset + i]

            # zero-fill ice fields when the timestep has no ice raster
            for f in ["ice_thick", "ice_vel", "on_ice"]:
                if f not in d or d[f] is None:
                    d[f] = 0

            # boolean regime flag so rf can learn glacial vs post-glacial directly
            d["ice_present"] = 1 if ka in ICE_TIMESTEPS else 0

            rows.append(d)
    return rows


# outer loop: for each ka, build presences, build pseudo-absences 1:1 to valid presences,
# collect rows into a growing list, write everything to csv at the end
def main():
    run_folder, out_gdb = get_run_folder()
    print(f"writing to {out_gdb}")

    reset_environment(REF_RASTER)
    sites_bng = project_sites(out_gdb)
    print(f"projected {arcpy.management.GetCount(sites_bng)[0]} sites")

    columns = ["site_name", "timestep_ka", "x", "y", "status", "ice_present",
               "date_from", "date_until", "date_span",
               "twi", "slope", "faf", "head_surf", "lake_bin", "lake_depth",
               "ice_thick", "ice_vel", "geology", "on_ice", "on_water",
               "dist_coast", "dist_lake", "dist_river"]

    all_rows = []
    for ka in TIMESTEPS:
        # re-pin env to this ka's twi so extractions honour the timestep's snap and extent
        reset_environment(os.path.join(TERRAIN_GDB, f"twi_{ka}ka"))

        pres_pts, n_pres = process_presence(sites_bng, ka, out_gdb)
        pres_rows = collect_rows(pres_pts, ka, True) if pres_pts else []
        n_valid = sum(1 for r in pres_rows if r["status"] == 1)
        n_99 = n_pres - n_valid

        # match pseudo-absence count to valid presences only, status-99 don't count
        n_bg = int(n_valid * BG_RATIO)
        if n_bg > 0:
            habitat = build_habitat(ka, for_background=True)
            bg_pts = process_background(habitat, pres_pts, n_bg, ka, out_gdb)
            bg_rows = collect_rows(bg_pts, ka, False)
        else:
            bg_rows = []

        all_rows += pres_rows + bg_rows
        # background can fall short of the target if the surveyed domain is small at this ka
        shortfall = "" if len(bg_rows) >= n_bg else f"  (target {n_bg}, short by {n_bg - len(bg_rows)})"
        print(f"  {ka}ka: {n_pres} presence ({n_valid} valid, {n_99} on ice/water), "
              f"{len(bg_rows)} background{shortfall}")

    csv_path = os.path.join(run_folder, "training_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"\nsaved {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()


