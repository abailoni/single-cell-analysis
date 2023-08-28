import os
import pandas as pd
import tifffile

metadata = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/initial_reports/metadata_mod_full_plus_MALDI_size.csv")

# "/scratch/bailoni/projects/spacem-reports/australia-project/initial_reports/metadata_mod_full_plus_MALDI_size.csv"

microscopy_dir = "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/initial_reports/microscopy"

tif_file_pattern = "{slide} well{well}{channel_type}.tif"

channel_specs = {
    "pre-maldi": [[], ""],
    "DAPI": [[], "H"],
    "post-maldi": [[], " PM"],
}

import numpy as np

# Open all tif images and stack them together in three 3D numpy arrays:
# - one for pre-maldi
# - one for DAPI
# - one for post-maldi
#
# The three arrays should have the same shape, and the shape should be
# (number_of_slides, number_of_wells, number_of_channels, height, width)

metadata.sort_values(by=["slide", "well"], inplace=True)

# Loop over all files in the microscopy directory:
for index, row in metadata.iterrows():
    for channel_type in channel_specs:
        tif_file = tif_file_pattern.format(slide=row["slide"], well=row["well"], channel_type=channel_specs[channel_type][1])
        img_path = os.path.join(microscopy_dir, tif_file)
        if os.path.exists(img_path):
            # Load the tif image and append to the list of tif files for this channel type:
            channel_specs[channel_type][0].append(np.array(tifffile.imread(img_path))[..., 0]) # We only take the first channel of the tif image
            print(f"Success: {row['slide']} {row['well']} {channel_type}")
        else:
            print(f"File not found: {row['slide']} {row['well']} {channel_type}")


# Some of the images may have different shapes, so we need to find first the max shape and then pad all of them to that shape:
# keep in mind that every list channel_specs[channel_type][0] contains a list of arrays of different shapes!
max_shape = np.max([np.max([channel.shape for channel in channel_specs[channel_type][0]], axis=0) for channel_type in channel_specs], axis=0)

# Pad and Stack the tif images together:
for channel_type in channel_specs:
    channel_specs[channel_type][0] = np.stack([np.pad(channel, ((0, max_shape[0] - channel.shape[0]), (0, max_shape[1] - channel.shape[1]), (0, max_shape[2] - channel.shape[2])), mode="constant") for channel in channel_specs[channel_type][0]], axis=1)

zarr_out_group = "/scratch/bailoni/datasets/australia_project/shared_data_zarr"

# Remove the zarr group if it already exists:
import shutil
if os.path.exists(zarr_out_group):
    shutil.rmtree(zarr_out_group)
# Create the zarr group:
os.makedirs(zarr_out_group, exist_ok=True)

# Now save the three 3D numpy arrays to disk as zarr files (one zarr, three subgroups, use ome-zarr format):
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


for channel_type in channel_specs:
    zarr_out_path = os.path.join(zarr_out_group, channel_type)
    os.makedirs(zarr_out_path, exist_ok=True)
    # write the image data
    store = parse_url(zarr_out_path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=channel_specs[channel_type][0], group=root, axes="zyx",
                storage_options=dict(chunks=(1, max_shape[0], max_shape[1]))
                )

    # zarr_out = zarr.open(zarr_out_path, mode="w")
    # zarr_out.create_dataset("data", data=channel_specs[channel_type][0], chunks=(1, 1, 1, max_shape[0], max_shape[1], max_shape[2]), compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2), dtype=np.uint16, shape=(metadata["slide"].nunique(), metadata["well"].nunique(), max_shape[0], max_shape[1], max_shape[2]))

import neuroglancer
dimensions=neuroglancer.CoordinateSpace(
                scales=[1, 1, 1],
                units=['', '', ''],
                names=['z', 'y', 'x'])


n_layer = neuroglancer.ImageLayer

with viewer.txn() as s:
    for channel_type in channel_specs:
        s.layers[channel_type] = n_layer(
                source=neuroglancer.LocalVolume(
                    data=np.array(channel_specs[channel_type][0]),
                    dimensions=dimensions,
                )
            )
