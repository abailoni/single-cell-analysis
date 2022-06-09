import numpy as np
import cv2
import tifffile
import shutil

# to_process = {5: [4,7,8],
#               6: [3,4,7]}
# to_process = {5: [3]}
to_process = {6: [8]}


DWS_FACTOR = 2

pattern = "/scratch/bailoni/projects/gastrosome_processing_full/spacem/slide{slide}/W{well}/PreMaldi/" \
          "A1.hdf5_fused_tp_0_ch_{channel}.tif"
for slide in to_process:
    for well in to_process[slide]:
        for channel in [0, 1]:
            img_path = pattern.format(slide=slide,
                                      well=well,
                                      channel=channel)
            print("Reading {}-{}-{}...".format(slide, well, channel))
            img = tifffile.imread(img_path)
            print(img.shape)

            print("Resizing...")
            img = cv2.blur(img, (DWS_FACTOR, DWS_FACTOR))
            img = img[::DWS_FACTOR, ::DWS_FACTOR]
            # img

            # Backup:
            print("Backup and write...")
            shutil.copy(img_path, img_path.replace(".tif", "_BAK.tif"))
            tifffile.imwrite(
                img_path, img,
                # , ome=True
            )
