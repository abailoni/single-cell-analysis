import numpy as np
import pandas as pd
import spatialdata


sdata = spatialdata.read_zarr("/scratch/bailoni/datasets/australia_project/data_2024/neurons_april_24/spacem_v2/spatialdata.zarr")
layout_4_geo_df = sdata.shapes["slide4.layout"]
layout_4_df = pd.DataFrame([polygon.bounds for polygon in layout_4_geo_df.geometry], columns=["x_min", "y_min", "x_max", "y_max"])
maldi_regions_4_geo_df = sdata.shapes["slide4.maldi_regions"]
maldi_regions_4_df = pd.DataFrame([polygon.bounds for polygon in maldi_regions_4_geo_df.geometry], columns=["x_min", "y_min", "x_max", "y_max"])

layout_4_geo_df.geometry.overlaps(maldi_regions_4_geo_df.geometry)
# [(0, True), (1, False), (2, False), (3, False), (4, False), (5, False), (6, False), (7, False), (8, False), (9, False)]
