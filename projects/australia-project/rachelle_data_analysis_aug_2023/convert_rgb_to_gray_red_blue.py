import dask.array as da
import dask.array.image
from pathlib import Path
from tifffile import imwrite
import sys


def separate_rgb_to_gray_red_blue(
    input_path: Path, output_path: Path
):
    """
    Converts an RGB file to cyx with gray (brightfield) and red and blue stains as
    separated channels.
    """
    array = dask.array.image.imread(input_path)
    if array.ndim == 4:
        assert array.shape[0] == 1
        array = array[0]
    assert array.ndim == 3 and array.shape[-1] == 3
    r, g, b = array.transpose((2, 0, 1))
    # No stain has a green component, so green equals gray.
    gray = g
    # Red and blue channels contain stain + gray component
    red_stain = da.abs(r - gray)
    blue_stain = da.abs(b - gray)
    cyx = da.stack([gray, red_stain, blue_stain])
    # Warning: Not using Dask, loads the whole image to memory!
    imwrite(
        output_path,
        cyx,
        imagej=True,
        metadata={
            "axes": "CYX",
        },
    )



if __name__ == "__main__":
    separate_rgb_to_gray_red_blue(sys.argv[1], sys.argv[2])
    #separate_rgb_to_gray_red_blue(
    #    "/media/andreas/EMBL-SpaceM1/spacem-ht-datasets/australia_project/data_rachelle_aug_23/shared_data/Mushroom_Ox_Stress/PreMaldi/20230809_SH SY5Y_Mushroom and ox stress_Slide 1.tif",
    #    "/media/andreas/EMBL-SpaceM1/spacem-ht-datasets/australia_project/data_rachelle_aug_23/shared_data/Mushroom_Ox_Stress/PreMaldi_t2/20230809_SH SY5Y_Mushroom and ox stress_Slide 1.tif",
    #)

