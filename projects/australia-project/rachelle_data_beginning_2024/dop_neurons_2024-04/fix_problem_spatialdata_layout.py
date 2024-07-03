from pathlib import Path

import spatialdata

from spacem_lib.datastructures.slide_identifier import SlideIdentifier
from spacem_modules.input_conversion.layout import convert_layout_file_to_spatialdata_shapes

sdata = spatialdata.read_zarr("/home/jovyan/shared_bailoni/aus/proc/neurons/spacem_v2/spatialdata.zarr")
# sdata = spatialdata.read_zarr("/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v2/spatialdata.zarr")
# layout_path = Path("/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v2/10-well-chamber-slide-australia-project.json")
layout_path = Path("/home/jovyan/shared_bailoni/aus/proc/neurons/microscopy/10-well-chamber-slide-australia-project.json")
for slide_id in ("slide4", "slide5", "slide6"):
    identifier = SlideIdentifier(project_id="13_DA_neurons_11291_and_11574_d50_Cellbright_green_and_hoechst_pre_MSI", slide_id=slide_id)
    convert_layout_file_to_spatialdata_shapes(
        sdata=sdata,
        layout_path=layout_path,
        layout_name=f"{slide_id}.layout",
        identifier=identifier,
    )
