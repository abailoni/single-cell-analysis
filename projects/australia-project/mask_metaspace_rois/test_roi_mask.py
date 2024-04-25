# Load csv:
import pandas as pd

# Load a anndata file as example:
import scanpy as sc
import re

# Assuming 'big_data.csv' is your file
# Load only the column names from the CSV
df = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/mask_rois/test_rois.csv", nrows=0)

# Use a regular expression to extract (x, y) pairs from all column names
pattern = r"ROI \d+_x(\d+)_y(\d+)"
xy_df = df.columns.str.extract(pattern)
xy_df.dropna(inplace=True)  # Drop rows where no matches were found

# Convert extracted string values to integers
xy_df = xy_df.astype(int)
xy_df.columns = ['x', 'y']


adata = sc.read_h5ad(
    "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/march_astrocytes/metaspace_adata/4-A1.h5ad")


# Example DataFrame to be filtered
data = adata.obs

# Merging data DataFrame with xy_df to filter rows that match the (x, y) coordinates
filtered_data = pd.merge(data, xy_df, on=['x', 'y'], how='inner')

# TODO: plan:
#  - I need a first function that takes the dataframe including the xy columns, extracts the coordinate pairs, and just returs a 2D mask numpy matrix from given coordinates and also the coordinates as a dataframe
#  - Then I also need a second function that uses the first function that does the full filtering on all adatas for PixelQC report:
#         - Load metadata csv file with the list of METASPACE dataset to process
#         - Read csv from a given folder, using METASPACE dataset names in the name of the csv file (names following the scheme '{METASPACE_dataset_name}_ROI.csv`), placing all coordinates in a single dataframe and specifying the dataset name in a column
#         - Load AnnDatas in batch from a given folder (all datasets in a single adata object)
#         - Filter adata.obs using the coordinates from the dataframe



