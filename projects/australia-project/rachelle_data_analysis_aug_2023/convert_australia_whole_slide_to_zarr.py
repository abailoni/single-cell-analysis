#! /usr/bin/python3

import json
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import typer
from dask_image.imread import imread
from dask_image.ndinterp import affine_transform
from ngff_writer.constants import DIMENSION_AXES, DIMENSION_SEPARATOR
from ngff_writer.writer import open_ngff_collections
from PIL import Image
from skimage.transform import AffineTransform, SimilarityTransform
from typing import Literal, Optional, Union
from spacem_lib.utils.dimension_utils import convert_matrix_dims

# Avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


FLUORESCENCE_CHANNELS = {"H", "GFP", "WGA", "DAPI"}


def convert_australia_whole_slide_to_ngff_zarr(
    tiff_path: Path,
    layout_path: Path,
    modality_name: str,  # Literal["pre_maldi", "post_maldi"],
    output_path: Path,
    pixel_size: Optional[float] = None,
    point_pairs_file: Optional[Path] = None,
    channel_names: Optional[list[str]] = None,
):
    """
    Convert a whole-slide image file into per-well Zarr files.

    This script produces similar output as SpaceM-Stitcher without stitching tiles.
    The whole-slide image is optionally registered and split into per-well images that are aligned
    with the layout.

    Usage:
    1. Run for one modality (e.g. `post_maldi`) without point pairs (but pixel size if known).
    2. Open it in Napari. It is displayed in physical units (layout coordinate system). Read target
       coordinates (micrometers) of at least 3 image features and match the corresponding
       coordinates (pixels) from the source image (the other modality) and enter the coordinate
       pairs into a CSV file.
    3. Run for the other modality with the created point pairs file.

    Args:
        tiff_path: A stitched whole-slide TIFF image
        layout_path: Path to the slide layout file
        modality_name: The name of the image modality
        pixel_size: Pixel size in micrometer/pixel. If not provided and no point pairs file is
            provided, the image is registered to the layout bounds.
        output_path: Path to the Zarr file to write (microscopy.zarr)
        point_pairs_file: Path to a CSV file containing point pairs in rows with columns
            `source_x`, `source_y`, `target_x`, `target_y` where source is in the image's pixel
            coordinates and target is the layout's physical coordinates.
        channel_names: Sequence of channel names. If not provided, channels are named `channel_0`
            etc.
    """
    image = imread(tiff_path)
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim == 3:
        if image.shape[-1] == 3:
            raise ValueError(
                "Image seems to be RGB with axis order YXC. "
                "An image with stains as separate channels in CYX order is required."
            )
    if output_path.suffix.lower() == ".zarr":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Assume directory
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "microscopy.zarr"
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(image.shape[0])]
    layout = json.loads(layout_path.read_text())
    if point_pairs_file is not None:
        point_pairs_df = pd.read_csv(point_pairs_file)
        if not {"source_x", "source_y", "target_x", "target_y"} <= set(
            point_pairs_df.columns
        ):
            raise ValueError(
                "Point pairs file must contain columns source_x, source_y, target_x, target_y"
            )
        if not len(point_pairs_df) >= 3:
            raise ValueError("At least 3 points required")
        image_to_layout = AffineTransform(dimensionality=2)
        image_to_layout.estimate(
            point_pairs_df[["source_y", "source_x"]],
            point_pairs_df[["target_y", "target_x"]],
        )
        pixel_size = np.mean(image_to_layout.scale)
    else:
        if pixel_size is not None:
            image_to_layout = SimilarityTransform(dimensionality=2, scale=pixel_size)
        else:
            well_bounds = np.array([well["bbox"] for well in layout["wells"]])
            layout_bounds = (
                np.min(well_bounds, axis=(0, 1)),
                np.max(well_bounds, axis=(0, 1)),
            )
            layout_shape = np.abs(layout_bounds[1] - layout_bounds[0])
            scale = np.mean(layout_shape / np.asarray(image.shape[-2:]))
            image_to_layout = SimilarityTransform(
                dimensionality=2, translation=layout_bounds[0], scale=scale
            )
            pixel_size = scale
    assert channel_names is not None
    assert image_to_layout is not None  # px → µm
    assert pixel_size is not None

    ngff_file = open_ngff_collections(
        store=output_path, dimension_separator=DIMENSION_SEPARATOR
    )

    for well in layout["wells"]:
        top_left, bottom_right = well["bbox"]
        # px → µm
        well_image_to_well = SimilarityTransform(scale=pixel_size)
        # µm → µm
        well_to_layout = SimilarityTransform(translation=np.asarray(top_left))
        # px → µm
        well_image_to_layout = well_image_to_well + well_to_layout
        # px → px
        well_image_to_whole_image = well_image_to_layout + image_to_layout.inverse
        # µm
        well_shape = np.abs(np.asarray(bottom_right) - top_left)
        # px
        output_shape = tuple(np.round(well_shape / pixel_size).astype(int))
        # Transform raster image to registered raster image
        image_transformed = da.stack(
            [
                affine_transform(
                    channel_image,
                    well_image_to_whole_image.params,
                    output_shape=output_shape,
                    prefilter=False,
                    order=0,
                ).reshape(output_shape)
                for channel_image in image
            ]
        )

        # Add well image to NGFF file
        ngff_collection = ngff_file.add_collection(well["well_id"])
        # CYX to TCZYX
        image5d = image_transformed[np.newaxis, :, np.newaxis, :, :]

        matrix5d = convert_matrix_dims(
            matrix=well_image_to_layout.params,  # well image to layout?
            dims_in="yx",
            dims_out="tczyx",
            homogeneous=True,
        )
        transform_dct = {"type": "affine", "parameters": matrix5d.tolist()}

        ngff_collection.add_image(
            image_name=modality_name,
            array=image5d,
            dimension_axes=("t", "c", "z", "y", "x"),
            channel_names=channel_names,
            transformation=transform_dct,
            channel_metadata=[
                {
                    "blending": "additive"
                    if channel_name in FLUORESCENCE_CHANNELS
                    else "translucent",
                    **determine_channel_contrast_limits(image5d[:, channel_index, ...]),
                }
                for channel_index, channel_name in enumerate(channel_names)
            ],
        )


def determine_channel_contrast_limits(
    image: Union[np.ndarray, da.Array], zero_start=True
) -> dict:
    start = 0 if zero_start else np.asarray(image.min()).item()
    end = np.asarray(image.max()).item()
    # Contrast limits must be monotonically increasing
    if end == start:
        end = start + 1
    return {"window": {"start": start, "end": end}}


if __name__ == "__main__":
    # convert_australia_whole_slide_to_ngff_zarr(
    #     tiff_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/R0-0_c00-00_s0-0_w1_t00_z00.pre_maldi.small.tif"
    #     ),
    #     layout_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/10-well-chamber-slide-australia-project.json"
    #     ),
    #     modality_name="pre_maldi",
    #     output_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/test_stitching_tool_output"
    #     ),
    #     pixel_size=6.418,
    #     point_pairs_file=None,
    #     channel_names=None,
    # )
    # # post_maldi_point_pairs:
    # # source_x,source_y: coordinates on R0-0_c00-00_s0-0_w1_t00_z00.post_maldi.small.tif
    # # target_x,target_y: coordinates on /scratch/eisenbar/australia_project/test_stitching_tool2/test_stitching_tool_output/microscopy.zarr
    # convert_australia_whole_slide_to_ngff_zarr(
    #     tiff_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/R0-0_c00-00_s0-0_w1_t00_z00.post_maldi.small.tif"
    #     ),
    #     layout_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/10-well-chamber-slide-australia-project.json"
    #     ),
    #     modality_name="post_maldi",
    #     output_path=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/test_stitching_tool_output"
    #     ),
    #     pixel_size=None,
    #     point_pairs_file=Path(
    #         "/scratch/eisenbar/australia_project/test_stitching_tool2/post_maldi_point_pairs.csv"
    #     ),
    #     channel_names=None,
    # )

    typer.run(convert_australia_whole_slide_to_ngff_zarr)
