import pandas as pd
import re

# Since I can't directly read the csv from an image, I'll simulate the dataframe structure from the provided image.
# This is a sample structure based on the image the user has uploaded.

df = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/neurons_UOW/UOW_neurons.csv")

# We need to define a regex pattern that will match the datasetName format and extract the required parts
pattern = r"Slide (?P<slide_number>\d+) (?P<row>[A-Z])(?P<col>\d)(?P<ROI_number>\d*)"

# If ROI_number is missing in the string, it should default to '0', which we can handle after extraction

# Apply the regex to the 'datasetName' column and expand the results into a new DataFrame
extracted_data = df['datasetName'].str.extract(pattern)

# Fill missing ROI_number values with '0'
extracted_data['ROI_number'] = extracted_data['ROI_number'].replace('', '0')

# Add the extracted data to the original DataFrame
df = pd.concat([df, extracted_data], axis=1)

#  Save the updated DataFrame to a new CSV file
df.to_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/g_shared/shared/alberto/projects/spacem-reports/australia-project/2024/neurons_UOW/UOW_neurons_updated.csv", index=False)
