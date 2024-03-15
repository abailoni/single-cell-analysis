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

import re
import warnings
from collections.abc import Hashable,Iterable
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
from skimage.util import img_as_ubyte

from ngff_writer.constants import DIMENSION_AXES, DIMENSION_SEPARATOR
from ngff_writer.writer import open_ngff_collections
from PIL import Image


def determine_channel_contrast_limits(image: Union[np.ndarray, da.Array], zero_start=True) -> dict:
    start = 0 if zero_start else np.asarray(image.min()).item()
    end = np.asarray(image.max()).item()
    return {"window": {"start": start, "end": end}}



def convert_lif_australia_file_to_ngff_zarr(
    input_lif_file: Path,
    out_dir: Path,
):
    out_dir.mkdir(exist_ok=True, parents=True)
    # out_dir = out_dir / "converted_tif_files"
    # out_dir.mkdir(exist_ok=True, parents=True)
    lif_file = LifFile(str(input_lif_file))

    # Regular expression pattern to match the slide number and well number
    pattern = r"Slide (\d+)/([A-Z]\d+) merged"

    for image_idx, image in enumerate(lif_file.get_iter_image()):
        # Find all matches of the pattern in the input sequence
        matches = re.findall(pattern, image.name)
        if not matches:
            continue
        # Assert only one match is found
        assert len(matches) == 1
        # Get the slide number and well number from the match
        slide_number, well_number = matches[0]

        out_path = out_dir / f"slide_{slide_number}/microscopy.zarr"
        out_path.mkdir(exist_ok=True, parents=True)

        ngff_file = open_ngff_collections(
            store=out_path,
            dimension_separator=DIMENSION_SEPARATOR
        )

        channel_colors = {
            "brightfield": "FFFFFF",
            "DAPI": "FF0000",
            "GFP": "00FF00",
        }
        channel_names = list(channel_colors)
        FLUORESCENCE_CHANNELS = ["DAPI", "GFP"]

        # Load channels:
        print(f"Loading img slide {slide_number}, well {well_number}...")
        channel_imgs = list(image.get_iter_c())
        channel_imgs = [np.asarray(img) for img in channel_imgs]
        tczyx_shape = (1, len(channel_imgs), 1) + channel_imgs[0].shape[-2:]
        pre_maldi_image = da.concatenate(channel_imgs).reshape(tczyx_shape)

        print("Writing....")
        ngff_collection = ngff_file.add_collection(well_number)
        ngff_collection.add_image(
            image_name="pre_maldi",
            array=pre_maldi_image,
            dimension_axes=DIMENSION_AXES,
            channel_names=channel_names,
            channel_metadata=[
                {
                    "color": color,
                    "blending": "additive" if channel_name in FLUORESCENCE_CHANNELS else "translucent",
                    **determine_channel_contrast_limits(
                        pre_maldi_image[:, channel_index, ...]),
                }
                for channel_index, (channel_name, color) in
                enumerate(zip(channel_names, channel_colors.values()))
            ],
        )

        # Clean up:
        del pre_maldi_image
        # Force garbage collection:
        gc.collect()


if __name__ == "__main__":
    # typer.run(convert_lif_australia_file_to_ngff_zarr)
    convert_lif_australia_file_to_ngff_zarr(
        input_lif_file=Path("/scratch/bailoni/datasets/australia_project/data_feb_24/data_20240102_mbAstro.lif"),
        out_dir=Path("/scratch/bailoni/datasets/australia_project/data_feb_24/ht_spacem_20240102_mbAstro")
    )
