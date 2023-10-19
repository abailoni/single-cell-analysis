import re
import warnings
from collections.abc import Hashable,Iterable
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np
import pims
import typer
from dask_image.imread import imread
from imageio import get_reader
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from ngff_writer.constants import DIMENSION_AXES, DIMENSION_SEPARATOR
from ngff_writer.writer import open_ngff_collections
from PIL import Image

# Avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

FILE_NAME_PATTERN = re.compile(r"^slide(?P<slide_id>[0-9])_.*_well_(?P<well_id>[A-Z][1-9])_(?P<channel_name>.*)\.tif$")

PRE_MALDI_CHANNELS = {"pre_maldi_ch1", "pre_maldi_ch0"}
POST_MALDI_CHANNELS = {"post_maldi_ch1", "post_maldi_ch0"}
FLUORESCENCE_CHANNELS = {"post_maldi_ch1", "pre_maldi_ch0"}


def convert_australia_project_to_ngff_zarr(
    shared_data_path: Path,
    output_path: Path,
):
    """
    Convert image data of a processed SpaceM 0.1 dataset into Zarr file format.

    Args:
        shared_data_path: Path to the shared dataset directory
        output_path: Path to the Zarr file to write
    """

    def extract_slide_well_id(path: Path) -> Union[tuple[str, str], tuple[None, None]]:
        m = re.match(FILE_NAME_PATTERN, path.name)
        if m:
            return m.group("slide_id"), m.group("well_id")
        return None, None

    tiff_files = sorted(Path(shared_data_path).glob("*.tif"))
    slide_ids = sorted(set(extract_slide_well_id(p)[0] for p in tiff_files))
    print(f"{slide_ids=}")
    ngff_files = {
        slide_id:
            open_ngff_collections(
                # store=output_path / f"{slide_id}.zarr", dimension_separator=DIMENSION_SEPARATOR
                store=output_path / f"microscopy.zarr", dimension_separator=DIMENSION_SEPARATOR
            )
        for slide_id in slide_ids if slide_id is not None
    }
    for (slide_id, well_id), group in groupby(tiff_files, key=extract_slide_well_id):
        well_tiff_files = list(group)
        if slide_id is None or well_id is None:
            warnings.warn(f"No slide_id or well_id detected for {list(well_tiff_files)}")
            continue
        # Sort the paths by channel names
        channel_names = [re.match(FILE_NAME_PATTERN, p.name).group("channel_name") for p in well_tiff_files]
        well_tiff_files, channel_names = zip(*sorted(zip(well_tiff_files, channel_names), key=itemgetter(1)))
        print(f"{slide_id=} {well_id=} {list(well_tiff_files)=} {channel_names=}") ############################################
        ngff_file = ngff_files[slide_id]
        ngff_collection = ngff_file.add_collection(well_id)
        pre_maldi_paths, pre_maldi_channel_names = zip(*[(p, n) for p, n in zip(well_tiff_files, channel_names) if n in PRE_MALDI_CHANNELS])
        # Possibly remove "pre_maldi_" from channel names:
        pre_maldi_channel_names = [n.replace("pre_maldi_", "") for n in pre_maldi_channel_names]
        post_maldi_paths, post_maldi_channel_names = zip(*[(p, n) for p, n in zip(well_tiff_files, channel_names) if n in POST_MALDI_CHANNELS])
        # Possibly remove "post_maldi_" from channel names:
        post_maldi_channel_names = [n.replace("post_maldi_", "") for n in post_maldi_channel_names]
        pre_maldi_image, pre_maldi_colors = _channel_paths_to_tczyx(pre_maldi_paths)
        post_maldi_image, post_maldi_colors = _channel_paths_to_tczyx(post_maldi_paths)

        ngff_collection.add_image(
            image_name="pre_maldi",
            array=pre_maldi_image,
            dimension_axes=DIMENSION_AXES,
            channel_names=pre_maldi_channel_names,
            channel_metadata=[
                {
                    "color": color,
                    "blending": "additive" if channel_name in FLUORESCENCE_CHANNELS else "translucent",
                    **determine_channel_contrast_limits(pre_maldi_image[:, channel_index, ...]),
                }
                for channel_index, (channel_name, color) in enumerate(zip(pre_maldi_channel_names, pre_maldi_colors))
            ],
        )
        ngff_collection.add_image(
            image_name="post_maldi",
            array=post_maldi_image,
            dimension_axes=DIMENSION_AXES,
            channel_names=post_maldi_channel_names,
            channel_metadata=[
                {
                    "color": color,
                    "blending": "additive" if channel_name in FLUORESCENCE_CHANNELS else "translucent",
                    **determine_channel_contrast_limits(post_maldi_image[:, channel_index, ...]),
                }
                for channel_index, (channel_name, color) in enumerate(zip(post_maldi_channel_names, post_maldi_colors))
            ],
        )


def determine_channel_contrast_limits(image: Union[np.ndarray, da.Array], zero_start=True) -> dict:
    start = 0 if zero_start else np.asarray(image.min()).item()
    end = np.asarray(image.max()).item()
    return {"window": {"start": start, "end": end}}


def all_equal(collection: Iterable[Hashable]) -> bool:
    """
    Returns True if all elements in a collection are equal.
    """
    return len(set(collection)) == 1


def _extract_channel_color_from_rgb_image(image) -> str:
    if image.ndim <= 2 or image.shape[-1] != 3:
        warnings.warn(f"Image is not RGB")
        return "FFFFFF"
    rgb = img_as_ubyte(np.asarray(np.max(np.reshape(image, (-1, 3)), axis=0)))
    r, g, b = rgb
    return f"{r:02X}{g:02X}{b:02X}"


def _channel_paths_to_tczyx(channel_paths: list[Path]) -> tuple[da.Array, list[str]]:
    if not channel_paths:
        raise ValueError("Image must have at least one channel")
    try:
        channel_imgs = list(imread(path) for path in channel_paths)
    except pims.api.UnknownFormatError:
        # imread(path) does not detect file types of image files without extension.
        # Dask image does not have a low-level API for detecting the file format, so fall-back to
        # imageio (no Dask, not chunked!).
        channel_imgs = list(get_reader(path).get_data(0) for path in channel_paths)
    colors = [_extract_channel_color_from_rgb_image(img) for img in channel_imgs]
    channel_imgs = [rgb2gray(img) for img in channel_imgs]
    if not all(img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1) for img in channel_imgs):
        raise ValueError(
            "SpaceM images must be single time-point, single channel, two-dimensional (y, x)."
        )
    if not all_equal(img.shape for img in channel_imgs):
        raise ValueError("All channels should have same size.")
    tczyx_shape = (1, len(channel_imgs), 1) + channel_imgs[0].shape[-2:]
    channels_img = da.concatenate(channel_imgs).reshape(tczyx_shape)
    return channels_img, colors


if __name__ == "__main__":
    typer.run(convert_australia_project_to_ngff_zarr)


