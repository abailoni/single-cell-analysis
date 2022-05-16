import json
import pandas as pd
import numpy as np


def get_molecules_names(am_matrix):
    var = am_matrix.var.copy()
    var.index = var.index.str.replace("[+-].*", "", regex=True)
    # var = var.loc[selected_metabolites]  # Subset to selected metabolites

    var = var.filter(regex=r"moleculeNames")  # Select all 'moleculeNames' columns
    var = var.applymap(json.loads, na_action="ignore")  # Parse json into Python lists

    # Gather molecule names across datasets (and databases)
    var = var.apply(
        lambda x: np.unique(
            np.array(
                x.dropna().tolist(),
                dtype=object
            )
        ),
        axis=1
    )

    var = var.drop_duplicates()
    var = var.explode()

    db_df = var.reset_index(name="name")
    db_df.index = pd.RangeIndex(start=1, stop=len(db_df) + 1, name="id")

    return var

    # print(f"Mapped {selected_metabolites.size} sum formulas to {len(db_df)} molecule names.")
