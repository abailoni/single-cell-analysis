from copy import deepcopy

from ngff_writer.datastructures import NgffImage
from ngff_writer.writer import open_ngff_image
from ngff_writer.writer_utils import add_image

from skimage.transform import AffineTransform

path_full = "/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v1/slide_4/microscopy.zarr/A1/pre_maldi"
path_pattern = "/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v1/slide_{slide_id}/microscopy.zarr/{row}{col}"


for slide_id in ["4","5", "6"]:
    for row in ["A", "B"]:
        for col in range(1, 6): 
            print(f"Processing slide {slide_id}, well {row}{col}...")
            path_pattern_filled = path_pattern.format(slide_id=slide_id, row=row, col=col)

            image_post: NgffImage = open_ngff_image(f"{path_pattern_filled}/post_maldi")
            # array_np = image.array(dimension_axes=("y", "x"),
            #                        channel_name="GFP")
            # array_dask = image.array(dimension_axes=("c", "y", "x"),
            #                          as_dask=True)

            post_transf: AffineTransform = image_post.transform("yx")

            post_transf.params[0,0] *= 1.978125
            post_transf.params[1,1] *= 1.978125

            image_pre: NgffImage = open_ngff_image(
                f"{path_pattern_filled}/pre_maldi")
            image_pre.set_transform(post_transf,
                                    dimension_axes="yx")
            # break
        # break
    # break

