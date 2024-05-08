import shutil

import pandas as pd
from pathlib import Path
from distutils.dir_util import copy_tree

project_dir = Path("/scratch/bailoni/datasets/australia_project/data_2024/astrocytes_march_24/")

# Load dataframe with dataset IDs:
df = pd.read_csv(project_dir / "astrocytes_all_datasets.csv")
df = df[df["facility"] == "EMBL"]
SLIDE_KEY = "slide"

dir_default_config_files = project_dir / "DEFAULT_CONFIG_FILES_SPACEM_v1"
root_spacem_folder = project_dir / "spacem_v1_rachelle_premaldi"

for slide_id in ["4", "5", "6"]:
    print(f"Processing slide {slide_id}...")
    spacem_project_dir = root_spacem_folder / f"slide{slide_id}"
    spacem_project_dir.mkdir(parents=True, exist_ok=True)

    # Backup well_metadata.csv in the spacem project directory:
    if (spacem_project_dir / "well_metadata.csv").exists():
        shutil.copy(spacem_project_dir / "well_metadata.csv", spacem_project_dir / "well_metadata_backup.csv")

    # Copy default config files:
    copy_tree(str(dir_default_config_files), str(spacem_project_dir))

    old_target_df = pd.read_csv(spacem_project_dir / "well_metadata_backup.csv")
    if "maldi.metaspace_dataset_id" in old_target_df.columns:
        old_target_df.drop(columns=["maldi.metaspace_dataset_id"], inplace=True)

    # Update well_metadata.csv with the correct dataset IDs:
    source_df = df[df[SLIDE_KEY] == int(slide_id)]
    source_df["well_id"] = source_df["row"].astype("str") + source_df["col"].astype("str")
    source_df = source_df[["well_id", "datasetId"]]
    source_df.sort_values("well_id", inplace=True)
    source_df.rename(columns={"datasetId": "maldi.metaspace_dataset_id"}, inplace=True)

    target_df = pd.merge(old_target_df, source_df, on="well_id", how="outer")
    target_df.to_csv(spacem_project_dir / "well_metadata.csv", index=False)

    # Finally, update the config file with the correct slide number:
    slide_config_file = spacem_project_dir / "slide_metadata.json"
    # Replace all "slide5" occurrences with "slide{slide_id}":
    with open(slide_config_file, "r") as f:
        slide_config = f.read()
    slide_config = slide_config.replace("slide5", f"slide{slide_id}")
    with open(slide_config_file, "w") as f:
        f.write(slide_config)
