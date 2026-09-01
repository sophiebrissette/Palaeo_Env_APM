#Habitability Reconstructions
#boolean overlay 

import arcpy
import os
from arcpy.sa import Con, IsNull, Raster

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False

# input locations, one gdb per upstream pipeline step
project_root = r"Project Directory"
coast_folder = os.path.join(project_root, "XXX")
ice_gdb = os.path.join(project_root, "run_XXX", "ice_flag.gdb")
lake_gdb = os.path.join(project_root, "run_XXX", "lake_binary.gdb")
channel_gdb = os.path.join(project_root, "run_XXX", "step4_bool_inputs.gdb")

# next run folder and output gdb
existing = [d for d in os.listdir(project_root) if d.startswith("run_") and d.split("_")[1].isdigit()]
run_num = max([int(d.split("_")[1]) for d in existing], default=0) + 1
run_folder = os.path.join(project_root, "run_{:03d}".format(run_num))
os.makedirs(run_folder)
arcpy.management.CreateFileGDB(run_folder, "step5_habitability_bool.gdb")
out_gdb = os.path.join(run_folder, "step5_habitability_bool.gdb")
print("output folder: " + run_folder)

TIMESTEPS = [31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]

# 12ka run as ice free, glacial extent applied separately as a mask later
#issues with LLS and Younger Dryas - this timestep would benefit from additional environmental reconstructions 
ICE_FREE = [12]

# reads an input layer as 1/0, zero-fills if the raster does not exist
def to_binary(path, zero):
    if not arcpy.Exists(path):
        print("  missing, zero-filled: " + os.path.basename(path))
        return zero
    r = Raster(path)
    return Con(IsNull(r), 0, Con(r > 0, 1, 0))

for ka in TIMESTEPS:

    # coastline sets the grid for everything else this timestep
    coast_path = os.path.join(coast_folder, "Coast_{}ka.tif".format(ka))
    arcpy.env.snapRaster = coast_path
    arcpy.env.extent = coast_path
    arcpy.env.cellSize = coast_path

    bin_coast = Con(Raster(coast_path) > 0, 1, 0)  # already 1 = inundated, 0 = land
    bin_coast.save(os.path.join(out_gdb, "bin_coast_{}ka".format(ka)))
    print("saved: bin_coast_{}ka".format(ka))

    zero = bin_coast * 0

    # ice forced to zero for ice free timesteps, otherwise read from step 3/4
    if ka in ICE_FREE:
        bin_ice = zero
        print("  ice free timestep, ice layer set to 0")
    else:
        bin_ice = to_binary(os.path.join(ice_gdb, "Binary_Ice_{}ka".format(ka)), zero)
    bin_ice.save(os.path.join(out_gdb, "bin_ice_{}ka".format(ka)))
    print("saved: bin_ice_{}ka".format(ka))

    # remaining water constraints
    bin_lake = to_binary(os.path.join(lake_gdb, "lake_bin_{}ka".format(ka)), zero)
    bin_lake.save(os.path.join(out_gdb, "bin_lake_{}ka".format(ka)))
    print("saved: bin_lake_{}ka".format(ka))

    bin_melt = to_binary(os.path.join(channel_gdb, "chan_melt_bool_{}ka".format(ka)), zero)
    bin_melt.save(os.path.join(out_gdb, "bin_melt_{}ka".format(ka)))
    print("saved: bin_melt_{}ka".format(ka))

    bin_fluv = to_binary(os.path.join(channel_gdb, "chan_fluvial_bool_{}ka".format(ka)), zero)
    bin_fluv.save(os.path.join(out_gdb, "bin_fluvial_{}ka".format(ka)))
    print("saved: bin_fluvial_{}ka".format(ka))

    # nested Con applies constraints in priority order
    # 0 habitable, 1 ice, 2 inundated, 3 lake, 4 melt channel, 5 fluvial
    habitability = Con(bin_ice == 1, 1,
                   Con(bin_coast == 1, 2,
                   Con(bin_lake == 1, 3,
                   Con(bin_melt == 1, 4,
                   Con(bin_fluv == 1, 5, 0)))))
    habitability.save(os.path.join(out_gdb, "habitability_bool_{}ka".format(ka)))
    print("saved: habitability_bool_{}ka".format(ka))

print("done, all layers in " + out_gdb)
