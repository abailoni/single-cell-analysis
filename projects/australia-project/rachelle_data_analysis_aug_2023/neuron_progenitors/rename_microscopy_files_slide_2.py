import os.path
import re
import shutil
from pathlib import Path

input_path = "/scratch/bailoni/datasets/australia_project/data_neurons_sept_23/shared_data_progenitor/slide2_microscopy"
output_path = Path(input_path).parent / "slide2_microscopy_renamed"
# Remove output path if it exists
if output_path.exists():
    shutil.rmtree(output_path)
output_path.mkdir(exist_ok=True, parents=True)

tif_files = sorted(Path(input_path).rglob("*.tif"))

well_conditions = {
    "291_HC": "A5",
    "291_rep_HC": "B5",
    "507_GBA": "A2",
    "450_HC": "A3",
    "328_GBA": "A4",
}
all_conditions = set(well_conditions.keys())

for file in tif_files:
    image_type = "pre_maldi" if "preMSI" in file.name else "post_maldi"
    # Get the condition from the file name:
    condition = next((cond for cond in all_conditions if cond in file.name), None)
    if condition is None:
        raise ValueError(f"Could not find condition in {file.name}")
    # Get the channel number from the file name, depending on how the name is ending:
    # Two options for ending are "..._ch00_SV.tif"  or "..._ch01_SV.tif"
    if file.name.endswith("_ch00_SV.tif"):
        channel_number = "0"
    elif file.name.endswith("_ch01_SV.tif"):
        channel_number = "1"
    else:
        raise ValueError(f"Could not find channel number in {file.name}")
    # Now compose the new name:
    new_name = f"slide2_{condition}_well_{well_conditions[condition]}_{image_type}_ch{channel_number}.tif"
    # Copy the file to the output path:
    print("Renaming ", file, " to ", new_name)
    shutil.copyfile(file, output_path / new_name)



