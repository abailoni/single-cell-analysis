import sys
import numpy as np

if __name__ == "__main__":
    # Assuming that sys.argv[1] is a folder, call separate_rgb_to_gray_red_blue for all tif files in that folder:
    corner_coords = np.array(eval(sys.argv[1]))

    out_json_string = f"""{{"origin": [{corner_coords[0,1]}, {corner_coords[0, 0]}], "size": [{corner_coords[2,1]-corner_coords[0,1]}, {corner_coords[2, 0]-corner_coords[0,0]}]}}
    """
    out_filename = "./post_maldi_cropped.crop_frame.json"

    with open(out_filename, "w") as f:
        f.write(out_json_string)

