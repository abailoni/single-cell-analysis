import pandas as pd
import re

# Since I can't directly read the csv from an image, I'll simulate the dataframe structure from the provided image.
# This is a sample structure based on the image the user has uploaded.

df = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/all_neurons/all_neurons.csv")

well_conditions = {
    # "Control": ["A1", "A3", "A5", "B2", "B4"],
    # "Diseased": ["A2", "A4", "B1", "B3", "B5"],
    "Row A": ["A1", "A2", "A3", "A4", "A5"],
    "Row B": ["B1", "B2", "B3", "B4", "B5"],
}

df["well_id"] = df["row"] + df["col"].astype(str)
df["neuron_condition"] = df["well_id"].apply(
    lambda x: "Row A" if x in well_conditions["Row A"] else "Row B"
)

#  Save the updated DataFrame to a new CSV file
df.to_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/all_neurons/all_neurons_test.csv", index=False)
