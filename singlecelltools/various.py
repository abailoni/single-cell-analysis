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


def get_inchi(adata, list_molecules_dbs, copy=True,
              name_molecude_id_col="moleculeIds", name_annotation_id_col="annotation_id",
              name_out_inchi_col="inchi"):
    """
    If copy is False, adata object is updated. If True, the updated adata.var dataframe is returned
    """

    new_df_data = {'annotation_id': [],
                   'mol_index': [],
                   'num_mols': [],
                   'mol_id': []}

    # Unroll lists in the adata.var dataframe:
    for _, row in adata.var.iterrows():
        ids = eval(row[name_molecude_id_col])
        new_df_data["mol_id"] += ids
        new_df_data["mol_index"] += range(len(ids))
        new_df_data["annotation_id"] += [row[name_annotation_id_col] for _ in range(len(ids))]
        new_df_data["num_mols"] += [len(ids) for _ in range(len(ids))]
        # idx_range = range(len(ids))
    new_df = pd.DataFrame(new_df_data)

    # Now merge with METASPACE molecule databases:
    combined_molecules_db = pd.concat(list_molecules_dbs)
    merged_df = pd.merge(new_df, combined_molecules_db, how="left", left_on="mol_id", right_on="id")

    # Finally, get inchi column in original format:
    var_df = adata.var.copy()
    inchi_lists = [None for _ in range(len(var_df))]
    for _, row in merged_df.iterrows():
        idx = var_df.index.get_loc(row[name_annotation_id_col])
        inchi = inchi_lists[idx]
        if inchi is None:
            inchi = [None for _ in range(row.num_mols)]
        inchi[row.mol_index] = str(row.inchi)
        inchi_lists[idx] = inchi

    var_df[name_out_inchi_col] = inchi_lists

    if copy:
        return var_df
    else:
        adata.var[name_out_inchi_col] = inchi_lists
        return adata


