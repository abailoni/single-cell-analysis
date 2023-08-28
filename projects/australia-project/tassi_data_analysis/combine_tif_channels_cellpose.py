import os
import numpy as np
import tifffile as tiff
import argparse

def merge_channels(green_path, red_path, output_folder, crop_slice: str = None):
    """

    Parameters
    ----------
    green_path
    red_path
    output_folder
    crop_slice: For example: "(slice(200,300), slice(100, 400))"

    Returns
    -------

    """
    # Load green and red channels
    green_channel = tiff.imread(green_path)
    red_channel = tiff.imread(red_path)

    # If there are three channels, keep only first one:
    if len(green_channel.shape) == 3:
        green_channel = green_channel[..., 0]
    if len(red_channel.shape) == 3:
        # Check the channel with max value and keep that one:
        red_channel = red_channel[..., np.argmax(np.max(red_channel, axis=(0, 1)))]

    if crop_slice is not None:
        # Convert crop_slice to tuple:
        crop_slice = eval(crop_slice)
        # Crop the channels:
        green_channel = green_channel[crop_slice]
        red_channel = red_channel[crop_slice]

        # Save cropped images for debugging:
        # as path, add _cropped to the file name:
        green_path = f"{os.path.splitext(green_path)[0]}_cropped.tif"
        red_path = f"{os.path.splitext(red_path)[0]}_cropped.tif"
        # Save the cropped images:
        tiff.imwrite(green_path, green_channel)
        tiff.imwrite(red_path, red_channel)

    # Make sure both channels have the same dimensions
    if green_channel.shape != red_channel.shape:
        raise ValueError("The dimensions of the two channels do not match.")

    # Combine channels to create an RGB image
    merged_image = np.stack([red_channel, green_channel, np.zeros_like(green_channel)], axis=-1)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the file name from the original green channel path
    file_name = os.path.splitext(os.path.basename(green_path))[0]

    # Save the merged image
    merged_path = os.path.join(output_folder, f"{file_name}_MERGED.tif")
    tiff.imwrite(merged_path, merged_image)

    print(f"Merge completed. The merged image is saved at: {merged_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge green and red channel images into an RGB image.")
    parser.add_argument("-g", "--green_path", type=str, required=True, help="Path to the green channel image.")
    parser.add_argument("-r", "--red_path", type=str, required=True, help="Path to the red channel image.")
    parser.add_argument("--crop", type=str, help="Crop the image to the specified dimensions.", default=None)
    args = parser.parse_args()

    output_folder = os.path.dirname(args.green_path)
    merge_channels(args.green_path, args.red_path, output_folder, args.crop)
