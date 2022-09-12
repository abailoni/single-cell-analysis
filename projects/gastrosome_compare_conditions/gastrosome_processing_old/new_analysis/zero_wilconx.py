import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import outer_spacem as osm
import os


sns.set(
    rc={
        "figure.figsize":(5, 5),
        "legend.frameon": False
    },
    style="ticks",
    context="talk"
)

well = "Drug_W8"
analysis_name = "v1"

main_dir = os.path.join("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome",
                       well, "reprocessing")

adata = sc.read(os.path.join(main_dir, "single_cell_analysis/spatiomolecular_adata.h5ad"))

# intracell_ions = pd.read_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/molecules_databases/reannotated/AB_Gastrosome_DrugW8_intra_ions_v2.tsv",
                             # sep="\t", index_col=0)

proj_dir = os.path.join(main_dir, "analysis", analysis_name)

cond_col = "cell_type"

adata.obs[cond_col] = np.where(adata.obs["max_intensity-Annotations"] > 0., "Gastrosomes", "Other cells")
adata.obs = adata.obs.astype({cond_col: "category"})

nb_marked_cells = (adata.obs[cond_col] == "Gastrosomes").sum()
total_nb_cells = adata.obs[cond_col].shape[0]
print("Gastrosomes: {}/{} cells".format(nb_marked_cells, total_nb_cells))


adata.raw = adata # keep raw values for diff. analysis


print("Cells before filtering:", adata.shape[0])
sc.pp.filter_cells(adata, min_genes=10)
print("Cells after filtering:", adata.shape[0])

print("Ions before filtering:", adata.shape[1])
sc.pp.filter_genes(adata, min_cells=200)
print("Ions after filtering:", adata.shape[1])

# sc.pp.normalize_total(adata)
sc.pp.normalize_total(adata, target_sum=1., key_added='tic')

from singlecelltools.nonzero_wilcoxon import nonzero_wilcoxon
nonzero_wilcoxon(adata,
    groupby=cond_col,
    use_raw=False,
    key_added="zero_rank_genes_groups")
