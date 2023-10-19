import dask.array as da
import dask.array.image
from pathlib import Path
from tifffile import imwrite
import sys


def separate_rgb_to_gray_red_blue(
    input_path: Path
):
    """
    Converts an RGB file to cyx with gray (brightfield) and red and blue stains as
    separated channels.
    """
    print(f"Converting {input_path.stem}...")
    array = dask.array.image.imread(str(input_path))
    # print(array.shape)
    if array.ndim == 4:
        assert array.shape[0] == 1
        array = array[0]
    assert array.ndim == 3 and array.shape[-1] == 3
    r, g, b = array.transpose((2, 0, 1))
    # No stain has a green component, so green equals gray.
    gray = g
    # Red and blue channels contain stain + gray component
    # red_stain = da.abs(r - gray)
    blue_stain = da.abs(b - gray)
    channels = [gray, blue_stain]
    out_dir = input_path.parent / "separated_channels"
    out_dir.mkdir(exist_ok=True, parents=True)

    for ch in range(2):
        # cyx = da.stack([gray, red_stain, blue_stain])
        # Warning: Not using Dask, loads the whole image to memory!
        imwrite(
            out_dir / f"{input_path.stem}_ch{ch}.tif",
            channels[ch],
            imagej=True,
            metadata={
                "axes": "YX",
            },
        )



if __name__ == "__main__":
    # Assuming that sys.argv[1] is a folder, call separate_rgb_to_gray_red_blue for all tif files in that folder:
    for tif_file in Path(sys.argv[1]).glob("*.tif"):
        separate_rgb_to_gray_red_blue(tif_file)
        # print(f"Converted {tif_file}")


