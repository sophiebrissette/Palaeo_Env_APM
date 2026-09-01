##rf training and prediction script
#final version
#includes true/false toggles for various predictors
#includes statistical testing 

import arcpy
from arcpy.sa import *
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support)
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

PROJECT_DIR = r"Define Path"

TRAINING_CSV = os.path.join(PROJECT_DIR, "run_XXX", "training_samples.csv") #this is the output of the SA script
LAKE_DEPTH_GDB = os.path.join(PROJECT_DIR, "run_XXX", "lake_depth.gdb")
GEOLOGY_GDB = os.path.join(PROJECT_DIR, "Outputs", "run_XXX", "geology.gdb")
DIST_GDB = os.path.join(PROJECT_DIR, "run_XXX", "dist_water.gdb")
ICE_MASK_GDB = os.path.join(PROJECT_DIR, "run_XXX", "ice_flag.gdb")
TERRAIN_GDB = os.path.join(PROJECT_DIR, "run_XXX", "step4c_twi_slope.gdb")
HYDRO_GDB = os.path.join(PROJECT_DIR, "run_XXX", "step4b_meltwater_lakes.gdb")
LAKE_GDB = os.path.join(PROJECT_DIR, "run_XXX", "lake_binary.gdb")
ICE_GDB = os.path.join(PROJECT_DIR, "run_007", "ice_layers.gdb")
HABITABILITY_GDB = os.path.join(PROJECT_DIR, "run_106", "step5_habitability_bool.gdb")

# raster name pattern in ICE_MASK_GDB, only read if training ice_present varies per cell
ICE_FLAG_NAME = "Binary_Ice_{ka}ka"

TIMESTEPS = [ 10, 11, 12, 13, 14, 15 ]
ICE_TIMESTEPS = set(range(15, 32))

# set parameters

# any layer at 500m bng works as the env anchor, twi_30ka is arbitrary
DOMAIN_KA = 30

SPLIT_RATIOS = [0.5, 0.7, 0.9]

# one fit per seed per ratio, the sd across seeds is the uncertainty on each reported metric
SPLIT_SEEDS = [42, 7, 13, 88, 101]

# grouping cell for the split, matches the sample grid
GROUP_CELL = 500

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "oob_score": True,
    "n_jobs": -1,
}

DEFAULT_THRESHOLD = 0.5     # youden's j is computed per split alongside this

# equal probability intervals, six classes to parallel the deterministic habitability raster
CLASS_CUTS = (0.167, 0.333, 0.5, 0.667, 0.833)

FINAL_MODEL_MODE = "all_data"     # "all_data" or "primary_split"
PRIMARY_SPLIT = 0.7

# set False to regenerate the split table without redoing 26 prediction surfaces
RUN_PREDICTION = True

# bgs coverage stops at the modern coastline, so geology is missing for most offshore
# background points and cannot be used without biasing the sample
#toggle on for terrestrial case studies or once geological data is updated
INCLUDE_GEOLOGY = False

# dist_coast tracks the palaeocoastline directly, so it may be restating the inundation
# model rather than adding site preference, dist_lake and dist_river stay in either way
INCLUDE_DIST_COAST = True

# head_surf is consistently a top predictor but shreve hydraulic potential scales with
# elevation, so it may be encoding low ground rather than hydrology
INCLUDE_HEAD_SURF = False

GEOLOGY_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99]

# step5 classes: 0 habitable, 1 ice, 2 inundated, 3 lake, 4 melt channel, 5 fluvial
# keep everything except ice and inundated to match sample_assembly's background domain
HABITABLE_VALUES = [0, 3, 4, 5]

# canonical order so training and prediction columns cannot drift apart
ALL_CONTINUOUS = ["twi", "slope", "faf", "head_surf", "lake_depth", "ice_thick", "ice_vel",
                  "dist_coast", "dist_lake", "dist_river"]
BOOLEAN = ["lake_bin", "ice_present"]

def continuous_cols():
    drop = set()
    if not INCLUDE_HEAD_SURF:
        drop.add("head_surf")
    if not INCLUDE_DIST_COAST:
        drop.add("dist_coast")
    return [c for c in ALL_CONTINUOUS if c not in drop]

CONTINUOUS = continuous_cols()

# set by load_data, controls whether prediction reads the ice flag raster or fills with 1
ICE_PRESENT_PER_CELL = None

# end of parameters

def get_run_folder():
    existing = [d for d in os.listdir(PROJECT_DIR) if d.startswith("run_") and d[4:].isdigit()]
    next_num = max([int(d[4:]) for d in existing], default=0) + 1
    run_folder = os.path.join(PROJECT_DIR, f"run_{next_num:03d}")
    os.makedirs(run_folder)
    arcpy.management.CreateFileGDB(run_folder, "predictions.gdb")
    return run_folder, os.path.join(run_folder, "predictions.gdb")

def check_paths():
    missing = []
    for ka in TIMESTEPS:
        checks = [os.path.join(TERRAIN_GDB, f"twi_{ka}ka"),
                  os.path.join(TERRAIN_GDB, f"slope_{ka}ka"),
                  os.path.join(HYDRO_GDB, f"flow_acc_fluvial_{ka}ka"),
                  os.path.join(HYDRO_GDB, f"head_surface_{ka}ka"),
                  os.path.join(LAKE_GDB, f"lake_bin_{ka}ka"),
                  os.path.join(LAKE_DEPTH_GDB, f"lake_depth_{ka}ka"),
                  os.path.join(HABITABILITY_GDB, f"habitability_bool_{ka}ka")]
        dist_layers = ["lake", "river"] + (["coast"] if INCLUDE_DIST_COAST else [])
        checks += [os.path.join(DIST_GDB, f"dist_{w}_{ka}ka") for w in dist_layers]
        if ka in ICE_TIMESTEPS:
            checks.append(os.path.join(ICE_GDB, f"ice_combined_{ka}ka"))
            if ICE_PRESENT_PER_CELL:
                checks.append(os.path.join(ICE_MASK_GDB, ICE_FLAG_NAME.format(ka=ka)))
        missing += [p for p in checks if not arcpy.Exists(p)]

    if INCLUDE_GEOLOGY and not arcpy.Exists(os.path.join(GEOLOGY_GDB, "geology_litho")):
        missing.append(os.path.join(GEOLOGY_GDB, "geology_litho"))

    if missing:
        raise RuntimeError(f"{len(missing)} missing rasters:\n  " + "\n  ".join(missing))
    print(f"all rasters present for {len(TIMESTEPS)} timesteps")

def feature_names():
    features = CONTINUOUS + BOOLEAN
    if INCLUDE_GEOLOGY:
        features += [f"geol_{c}" for c in GEOLOGY_CLASSES]
    return features

def detect_ice_mode(df):
    # if training ice_present varies within a glaciated timestep it is a per-cell mask and prediction must
    #read the flag raster, if it is always 1 it is a timestep indicator and prediction should fill with 1 to match
    ice_rows = df[df["timestep_ka"].isin(ICE_TIMESTEPS)]
    if len(ice_rows) == 0:
        return False
    means = ice_rows.groupby("timestep_ka")["ice_present"].mean()
    per_cell = bool((means < 0.999).any())
    print(f"  ice_present is {'per-cell' if per_cell else 'constant per timestep'} in training")
    print(f"  glaciated timestep means: {means.round(3).to_dict()}")
    return per_cell

def load_data():
    global ICE_PRESENT_PER_CELL

    df = pd.read_csv(TRAINING_CSV)
    print(f"loaded {len(df)} rows")

    df = df[df["status"] != 99].copy()
    print(f"  {len(df)} after removing status=99")

    # geology only filters rows when it is actually a predictor
    required = CONTINUOUS + BOOLEAN
    if INCLUDE_GEOLOGY:
        df["geology"] = pd.to_numeric(df["geology"], errors="coerce")
        required = required + ["geology"]

    before = len(df)
    df = df.dropna(subset=required).copy()
    print(f"  {len(df)} after dropping rows with missing predictors ({before - len(df)} dropped)")

    if INCLUDE_GEOLOGY:
        df["geology"] = df["geology"].astype(int)
        unknown = set(df["geology"].unique()) - set(GEOLOGY_CLASSES)
        if unknown:
            print(f"  warning: geology values outside GEOLOGY_CLASSES, dropping: {sorted(unknown)}")
            df = df[df["geology"].isin(GEOLOGY_CLASSES)].copy()
        for cls in GEOLOGY_CLASSES:
            df[f"geol_{cls}"] = (df["geology"] == cls).astype(int)

    ICE_PRESENT_PER_CELL = detect_ice_mode(df)

    # one grid cell is one group, so every record at a location falls on the same side of the split 
    #regardless of timestep or class - site_name is not used, it repeats across unrelated locations in the ads export
    df["x_grid"] = (df["x"] // GROUP_CELL * GROUP_CELL).astype(int)
    df["y_grid"] = (df["y"] // GROUP_CELL * GROUP_CELL).astype(int)
    df["site_id"] = "loc|" + df["x_grid"].astype(str) + "|" + df["y_grid"].astype(str)

    n_loc = df["site_id"].nunique()
    n_pres_loc = df.loc[df["status"] == 1, "site_id"].nunique()
    n_bg_loc = df.loc[df["status"] == 0, "site_id"].nunique()
    print(f"  {n_loc} unique locations ({n_pres_loc} with presences, {n_bg_loc} with background)")
    print(f"  {n_pres_loc + n_bg_loc - n_loc} locations carry both classes")
    print(f"  mean records per location: {len(df) / n_loc:.2f}")

    n_pres = int((df["status"] == 1).sum())
    n_bg = int((df["status"] == 0).sum())
    print(f"  {n_pres} valid presences, {n_bg} pseudo-absences (ratio {n_pres/max(n_bg,1):.2f}:1)")

    # sample_assembly generates these 1:1, a skewed ratio means one class is dropping out on nodata more often than the other
    if n_bg and not 0.8 <= n_pres / n_bg <= 1.25:
        print(f"  note: class balance has drifted from the 1:1 sampling design")

    return df

def build_xy(df):
    return df[feature_names()].values, df["status"].values.astype(int)

def site_blocked_split(df, train_ratio, seed):
    rng = np.random.default_rng(seed)

    # locations are stratified by which classes they contain so the class ratio holds in 
    # both partitions without any location being split
    by_loc = df.groupby("site_id")["status"].agg(["min", "max"])
    strata = [by_loc.index[by_loc["min"] == 1],
              by_loc.index[by_loc["max"] == 0],
              by_loc.index[(by_loc["min"] == 0) & (by_loc["max"] == 1)]]

    train_locs = set()
    for locs in strata:
        shuffled = rng.permutation(np.asarray(locs))
        train_locs.update(shuffled[:int(len(shuffled) * train_ratio)])

    train_mask = df["site_id"].isin(train_locs)
    return df[train_mask].copy(), df[~train_mask].copy()

def youdens_j(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])

def confusion_report(y_true, y_prob, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  {label} (threshold={threshold:.3f})")
    print(f"                    pred absent    pred present")
    print(f"    actual absent   {tn:>7}        {fp:>7}")
    print(f"    actual present  {fn:>7}        {tp:>7}")
    print(f"    precision  absent {prec[0]:.3f}   presence {prec[1]:.3f}")
    print(f"    recall     absent {rec[0]:.3f}   presence {rec[1]:.3f}")
    print(f"    f1         absent {f1[0]:.3f}   presence {f1[1]:.3f}")

    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "prec_absent": prec[0], "prec_pres": prec[1],
            "rec_absent": rec[0], "rec_pres": rec[1],
            "f1_absent": f1[0], "f1_pres": f1[1]}

def evaluate_split(df, train_ratio, seed):
    print(f"\n{'='*60}")
    print(f"split: {train_ratio:.0%} train / {1-train_ratio:.0%} test, seed {seed}")
    print(f"{'='*60}")

    train_df, test_df = site_blocked_split(df, train_ratio, seed)
    print(f"  train {len(train_df)}: "
          f"{(train_df['status']==1).sum()} pres, {(train_df['status']==0).sum()} bg, "
          f"{train_df['site_id'].nunique()} locations")
    print(f"  test  {len(test_df)}: "
          f"{(test_df['status']==1).sum()} pres, {(test_df['status']==0).sum()} bg, "
          f"{test_df['site_id'].nunique()} locations")

    overlap = set(train_df["site_id"]) & set(test_df["site_id"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} locations appear in both partitions")

    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    test_acc = ((y_prob >= DEFAULT_THRESHOLD).astype(int) == y_test).mean()

    # oob runs above the location-blocked test score, the gap is spatial autocorrelation
    print(f"\n  AUC-ROC: {auc:.4f}")
    print(f"  OOB score: {rf.oob_score_:.4f}")
    print(f"  test accuracy: {test_acc:.4f}  (oob gap {rf.oob_score_ - test_acc:+.4f})")

    default_metrics = confusion_report(y_test, y_prob, DEFAULT_THRESHOLD, "default")
    j = youdens_j(y_test, y_prob)
    j_metrics = confusion_report(y_test, y_prob, j, "youden's j")

    summary = {"ratio": train_ratio, "seed": seed, "auc": auc, "oob": rf.oob_score_,
               "test_acc": test_acc, "oob_gap": rf.oob_score_ - test_acc,
               "j_threshold": j,
               "n_train": len(train_df), "n_test": len(test_df),
               "train_locs": train_df["site_id"].nunique(),
               "test_locs": test_df["site_id"].nunique()}
    for k, v in default_metrics.items():
        summary[f"def_{k}"] = v
    for k, v in j_metrics.items():
        summary[f"j_{k}"] = v
    return summary, rf

def train_final(df):
    print(f"\n{'='*60}")
    print(f"training final model on all data")
    print(f"{'='*60}")
    X, y = build_xy(df)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)
    print(f"  OOB score: {rf.oob_score_:.4f}")
    return rf

def save_importances(rf, run_folder):
    out = pd.DataFrame({"feature": feature_names(), "importance": rf.feature_importances_})
    out = out.sort_values("importance", ascending=False)
    path = os.path.join(run_folder, "feature_importances.csv")
    out.to_csv(path, index=False)

    print(f"\nfeature importances (top 10):")
    for _, row in out.head(10).iterrows():
        print(f"  {row['feature']:<20} {row['importance']:.4f}")
    print(f"saved full importances to {path}")

def load_predictor_stack(ka):
    grid_path = os.path.join(TERRAIN_GDB, f"twi_{DOMAIN_KA}ka")
    grid_ref = arcpy.Raster(grid_path)

    # env must be pinned before any read, map algebra output conforms to these settings
    arcpy.env.snapRaster = grid_path
    arcpy.env.cellSize = grid_path
    arcpy.env.extent = grid_ref.extent
    arcpy.env.outputCoordinateSystem = arcpy.Describe(grid_path).spatialReference
    arcpy.env.resamplingMethod = "NEAREST"   # geology and habitability are categorical

    target = (grid_ref.height, grid_ref.width)

    def to_arr(path):
        # Float() forces map algebra so the result lands on the env grid, a raw RasterToNumPyArray
        # read would return the source raster's own grid
        arr = arcpy.RasterToNumPyArray(Float(arcpy.Raster(path)),
                                       nodata_to_value=-9999).astype(np.float32)
        if arr.shape != target:
            raise RuntimeError(f"{path} returned {arr.shape}, expected {target}")
        arr[arr == -9999] = np.nan
        return arr

    twi = to_arr(os.path.join(TERRAIN_GDB, f"twi_{ka}ka"))
    slope = to_arr(os.path.join(TERRAIN_GDB, f"slope_{ka}ka"))
    faf = to_arr(os.path.join(HYDRO_GDB, f"flow_acc_fluvial_{ka}ka"))
    head_surf = to_arr(os.path.join(HYDRO_GDB, f"head_surface_{ka}ka"))
    lake_bin = to_arr(os.path.join(LAKE_GDB, f"lake_bin_{ka}ka"))
    lake_depth = to_arr(os.path.join(LAKE_DEPTH_GDB, f"lake_depth_{ka}ka"))

    shape = twi.shape

    dist_lake = to_arr(os.path.join(DIST_GDB, f"dist_lake_{ka}ka"))
    dist_river = to_arr(os.path.join(DIST_GDB, f"dist_river_{ka}ka"))

    if INCLUDE_DIST_COAST:
        dist_coast = to_arr(os.path.join(DIST_GDB, f"dist_coast_{ka}ka"))
    else:
        dist_coast = np.zeros(shape, dtype=np.float32)

    if ka in ICE_TIMESTEPS:
        ice_composite = os.path.join(ICE_GDB, f"ice_combined_{ka}ka")
        ice_thick = to_arr(os.path.join(ice_composite, "Band_1"))
        ice_vel = to_arr(os.path.join(ice_composite, "Band_3"))
        if ICE_PRESENT_PER_CELL:
            ice_present = to_arr(os.path.join(ICE_MASK_GDB, ICE_FLAG_NAME.format(ka=ka)))
            ice_present = np.where(np.isnan(ice_present), 0, ice_present).astype(np.float32)
        else:
            ice_present = np.ones(shape, dtype=np.float32)
    else:
        ice_thick = np.zeros(shape, dtype=np.float32)
        ice_vel = np.zeros(shape, dtype=np.float32)
        ice_present = np.zeros(shape, dtype=np.float32)

    # zero is the physically correct default for these, unlike the terrain layers
    np.nan_to_num(ice_thick, copy=False, nan=0.0)
    np.nan_to_num(ice_vel, copy=False, nan=0.0)
    lake_bin = np.where(np.isnan(lake_bin), 0, lake_bin)
    lake_depth = np.where(np.isnan(lake_depth), 0, lake_depth)

    # domain comes from the step5 habitability raster so the rf predicts on exactly the deterministic model's surface
    hab = to_arr(os.path.join(HABITABILITY_GDB, f"habitability_bool_{ka}ka"))
    valid_mask = np.isin(hab, HABITABLE_VALUES)

    # a single nodata hole in any one terrain layer removes the cell entirely
    for a in (twi, slope, faf, head_surf):
        valid_mask &= ~np.isnan(a)

    arrays = {"twi": twi, "slope": slope, "faf": faf, "head_surf": head_surf,
              "lake_depth": lake_depth, "ice_thick": ice_thick, "ice_vel": ice_vel,
              "dist_coast": dist_coast, "dist_lake": dist_lake, "dist_river": dist_river,
              "lake_bin": lake_bin, "ice_present": ice_present}

    features = [arrays[c] for c in CONTINUOUS] + [arrays[b] for b in BOOLEAN]
    if INCLUDE_GEOLOGY:
        geology = to_arr(os.path.join(GEOLOGY_GDB, "geology_litho"))
        for cls in GEOLOGY_CLASSES:
            features.append((geology == cls).astype(np.float32))

    return np.stack(features, axis=-1), valid_mask, shape

def predict_surface(rf, ka, out_gdb):
    stack, valid_mask, shape = load_predictor_stack(ka)

    grid_ref = arcpy.Raster(os.path.join(TERRAIN_GDB, f"twi_{DOMAIN_KA}ka"))
    ll = grid_ref.extent.lowerLeft
    cell = grid_ref.meanCellWidth

    valid_flat = stack[valid_mask]
    if len(valid_flat) == 0:
        print(f"    {ka}ka: no valid cells, skipping")
        return

    prob = rf.predict_proba(np.nan_to_num(valid_flat, nan=0.0))[:, 1]

    out_prob = np.full(shape, np.nan, dtype=np.float32)
    out_prob[valid_mask] = prob
    prob_ras = arcpy.NumPyArrayToRaster(out_prob, ll, cell, cell, value_to_nodata=np.nan)
    prob_ras.save(os.path.join(out_gdb, f"prob_{ka}ka"))

    # digitize keeps this independent of how many cuts are set, 0 stays as nodata
    out_class = np.zeros(shape, dtype=np.int16)
    out_class[valid_mask] = np.digitize(out_prob[valid_mask], CLASS_CUTS) + 1
    class_ras = arcpy.NumPyArrayToRaster(out_class, ll, cell, cell, value_to_nodata=0)
    class_ras.save(os.path.join(out_gdb, f"class_{ka}ka"))

    print(f"    {ka}ka: predicted {len(prob)} cells, "
          f"mean prob {prob.mean():.3f}, top class {(out_class == len(CLASS_CUTS) + 1).sum()}")

def main():
    run_folder, out_gdb = get_run_folder()
    print(f"writing to {run_folder}")

    # loaded first so check_paths knows whether the ice flag rasters are needed
    df = load_data()
    if len(df) == 0:
        raise RuntimeError("no rows left after filtering, check the input csv")

    if RUN_PREDICTION:
        check_paths()

    results = []
    primary_rf = None
    for ratio in SPLIT_RATIOS:
        for seed in SPLIT_SEEDS:
            summary, rf = evaluate_split(df, ratio, seed)
            results.append(summary)

            # first seed only, otherwise the last iteration silently wins
            if (FINAL_MODEL_MODE == "primary_split" and abs(ratio - PRIMARY_SPLIT) < 1e-6
                    and seed == SPLIT_SEEDS[0]):
                primary_rf = rf

    results_df = pd.DataFrame(results)
    summary_path = os.path.join(run_folder, "split_comparison.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\nsaved {len(results_df)} split runs to {summary_path}")

    # sd across seeds is the uncertainty to report alongside each mean
    agg = (results_df.groupby("ratio")[["auc", "oob", "test_acc", "oob_gap", "j_threshold"]]
           .agg(["mean", "std"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(os.path.join(run_folder, "split_seed_summary.csv"), index=False)

    print(f"\nseed means at each ratio ({len(SPLIT_SEEDS)} seeds):")
    for ratio, row in agg.iterrows():
        print(f"  {ratio:.0%}  AUC {row['auc_mean']:.4f} +/- {row['auc_std']:.4f}  "
              f"gap {row['oob_gap_mean']:.4f} +/- {row['oob_gap_std']:.4f}")

    if FINAL_MODEL_MODE == "all_data":
        final_rf = train_final(df)
    else:
        final_rf = primary_rf
        print(f"\nusing rf from the {PRIMARY_SPLIT:.0%} split, seed {SPLIT_SEEDS[0]}, for prediction surfaces")

    save_importances(final_rf, run_folder)

    # saved before prediction so a crash in the raster loop does not cost the fit
    model_path = os.path.join(run_folder, "rf_model.joblib")
    joblib.dump(final_rf, model_path)
    print(f"saved model to {model_path}")

    if not RUN_PREDICTION:
        print(f"\nRUN_PREDICTION is False, stopping after training")
        return

    print(f"\n{'='*60}")
    print(f"generating prediction surfaces")
    print(f"{'='*60}")
    failed = []
    for ka in TIMESTEPS:
        try:
            predict_surface(final_rf, ka, out_gdb)
        except Exception as e:
            print(f"    {ka}ka failed: {e}")
            failed.append(ka)

    if failed:
        print(f"\n{len(failed)} timesteps failed: {failed}")
    print(f"\nall done, outputs in {run_folder}")

if __name__ == "__main__":
    main()
