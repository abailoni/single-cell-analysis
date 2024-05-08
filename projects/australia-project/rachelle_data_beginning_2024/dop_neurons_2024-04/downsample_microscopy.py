#
from pathlib import Path
import zarr
from ngff_writer.writer import open_ngff_collections
import tifffile

from ngff_writer.constants import DIMENSION_AXES, DIMENSION_SEPARATOR

# root_directory = Path("/scratch/bailoni/datasets/australia_project/new_data_feb_2024/astrocytes_march_24/spacem_v1_final/slide5")
root_directory = Path("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/bailoni/datasets/australia_project/new_data_feb_2024/astrocytes_march_24/rachelle_premaldi/slide5")

#  Find all sub-directories named microscopy.zarr in the root directory
microscopy_zarr_directories = list(root_directory.glob("**/microscopy.zarr"))

ngff_file = open_ngff_collections(
    store=root_directory / "microscopy.zarr",
    dimension_separator=DIMENSION_SEPARATOR
)
well_id = "A1"
ngff_collection = ngff_file.collections[well_id]

for microscopy_type in ["pre_maldi", "post_maldi"]:
    ngff_image = ngff_collection.images[f"{well_id}/{microscopy_type}"]
    # Convert to a tiff image:
    tifffile.imwrite(root_directory / f"{microscopy_type}.tiff", ngff_image.array())


