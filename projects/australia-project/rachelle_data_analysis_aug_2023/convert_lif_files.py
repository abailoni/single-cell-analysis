import os.path
import re
import shutil
from pathlib import Path

import readlif
import numpy as np
import tifffile
from tifffile import imwrite
from readlif.reader import LifFile
import gc

path_out_txt = Path("/scratch/bailoni/datasets/australia_project/data_rachelle_aug_23/new_data/out.txt")
MAIN_DIR = Path("/scratch/bailoni/datasets/australia_project/data_rachelle_aug_23/new_data")
input_directories = [
    MAIN_DIR / "microscopy_cellBrite",
    MAIN_DIR / "mushroom_PreMaldi",
    MAIN_DIR / "mushroom_PostMaldi",
]
main_output_dir_path = MAIN_DIR / "converted_tif_files"
main_output_dir_path.mkdir(exist_ok=True, parents=True)

for input_dir_path in input_directories:
    lif_files = sorted(Path(input_dir_path).rglob("*.lif"))
    output_dir_paths = [main_output_dir_path /  f"{lif_file.parent.name}/{lif_file.stem}" for lif_file in lif_files]

    for lif_file_path, out_path in zip(lif_files, output_dir_paths):
        out_path.mkdir(exist_ok=True, parents=True)
        lif_file = LifFile(str(lif_file_path))

        last_valid_name = ""
        for image_idx, image in enumerate(lif_file.get_iter_image()):
            if "egion" not in image.name:
                last_valid_name = image.name + "_"
            output_path = out_path / f"{last_valid_name.replace(' ', '_')}{image.name.replace(' ', '_')}"
#             print(output_path)

            # Only consider stitched images:
            if image.dims_n[1] > 2000:
                if 10 in image.dims_n and image.dims_n[10]>1:
                    continue
                print(f"Image n.{image_idx}: {image.name}")
                print("Extracting it....")
                print(image.dims_n)
                print(image.channels)

                print("Output path:", output_path)
                channels = [
                    np.array(image.get_frame(c=idx_ch))
                    for idx_ch in range(image.channels)
                ]
                merged_image = np.stack(channels, axis=0)
                print(merged_image.shape)
                output_path.mkdir(exist_ok=True, parents=True)
                # Copy out.txt file to output directory:
                shutil.copyfile(path_out_txt, output_path / "out.txt")
                output_path = output_path / "R0-0_c00-00_s0-0_w1_t00_z00.tif"
                imwrite(
                    output_path,
                    merged_image,
                    imagej=True,
                    metadata={
                        "axes": "CYX",
                    },
                )
                # Clean up:
                del merged_image
                # Force garbage collection:
                gc.collect()
#             break
#         break
#     break




