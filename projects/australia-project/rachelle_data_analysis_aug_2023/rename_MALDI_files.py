import os.path
import re
import shutil
from pathlib import Path

input_path = "/scratch/bailoni/datasets/australia_project/data_rachelle_aug_23/new_data/MSI_Mushroom"
output_path = Path("/scratch/bailoni/datasets/australia_project/data_rachelle_aug_23/new_data/Mushrooms_OxidativeStress_UOW_pos")
# Remove output path if it exists
if output_path.exists():
    shutil.rmtree(output_path)
output_path.mkdir(exist_ok=True, parents=True)

imzml_files = sorted(Path(input_path).rglob("*.imzML"))

well_conditions = {
    "NT": ["A2", "B1", "D2", "C2"],
    "C": ["B2", "D1", "E2", "B1"],
    "VC": ["C2", "D2", "C1", "C1"],
    "R+6": ["A1", "A2", "B2", "E1"],
    "6": ["B1", "C1", "D1", "D2"],
    "R+H": ["C1", "B2", "A1", "A2"],
    "R": ["D1", "C2", "B1", "E2"],
    "H": ["D2", "E2", "E1", "D1"],
    "C+H": ["E2", "E1", "A2", "A1"],
    "C+6": ["E1", "A1", "C2", "B2"],
}

replacements = {
    "A": "1",
    "B": "2",
    "C": "3",
    "D": "4",
    "E": "5",
    "1": "A",
    "2": "B"
}

transformed_well_conditions = {
    condition: [
        replacements[well[1]] + replacements[well[0]] for well in wells
    ]
    for condition, wells in well_conditions.items()
}

"""
transformed_well_conditions = {
 'NT': ['1B', '2A', '4B', '3B'],
 'c': ['2B', '4A', '5B', '2A'],
 'VC': ['3B', '4B', '3A', '3A'],
 'R+6': ['1A', '1B', '2B', '5A'],
 '6': ['2A', '3A', '4A', '4B'],
 'R+H': ['3A', '2B', '1A', '1B'],
 'R': ['4A', '3B', '2A', '5B'],
 'H': ['4B', '5B', '5A', '4A'],
 'C+H': ['5B', '5A', '1B', '1A'],
 'C+6': ['5A', '1A', '3B', '2B']
}
"""

# FILE_NAME_PATTERN = re.compile(r"Slide_(?P<slide_number>[1-9])/(?P<polarity>Pos|Neg) mode/(?P<condition>"+ "|".join([re.escape(cond) for cond in well_conditions]) + ").+\.tif$")
FILE_NAME_PATTERN = re.compile(r"(?P<condition>"+"|".join([re.escape(cond) for cond in well_conditions])+")_pos_(?P<slide_number>[1-9])\.imzML$")

ibd_files = [str(path).replace(".imzML", ".ibd") for path in imzml_files]
assert all(os.path.isfile(path) for path in ibd_files)

data = [(m.group("slide_number"), m.group("condition") ) for m in [re.match(FILE_NAME_PATTERN, path.name) for path in imzml_files] if m is not None]

out_names = [f"Slide{slide_number}_well_{transformed_well_conditions[condition][int(slide_number)-1]}_{condition}" for slide_number, condition in data]

# Copy imzml and ibd files to output_path using out_names:
for imzml_file, ibd_file, out_name in zip(imzml_files, ibd_files, out_names):
    out_imzml_file = output_path / f"{out_name}.imzML"
    out_ibd_file = output_path / f"{out_name}.ibd"
    assert not os.path.isfile(out_imzml_file)
    assert not os.path.isfile(out_ibd_file)
    shutil.copyfile(imzml_file, out_imzml_file)
    shutil.copyfile(ibd_file, out_ibd_file)


