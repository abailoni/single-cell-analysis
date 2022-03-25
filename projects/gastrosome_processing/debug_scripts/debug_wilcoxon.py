import matplotlib.pyplot as plt

import scanpy as sc
import pandas as pd
import seaborn as sns
from outer_spacem.io import convert_name

import numpy as np
from pathlib import Path

from outer_spacem.pl import plot_distributions, plot_umap_top_n, volcano_plot

well = "Well_3"

if well == "Well_8":
    # -------- WELL 8 -----------------------:
    adata = sc.read(
        "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/gastrosome_processing/SpaceM_processing_new/Drug_W8/analysis/single_cell_analysis/spatiomolecular_adata.h5ad")  # assuming you only have 1 dataset

    # filter out extracell stuff
    intracell_ions = pd.read_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/AB_Gastrosome_DrugW8_intra_ions_v2.tsv",
                                 sep="\t", index_col=0)
    # adata = adata[:, adata.var.formula.isin(intracell_ions["name"])].copy()


    data_dir = Path(r"/Users/alberto-mac/EMBL_ATeam/projects/gastrosome")

    proj_dir = "Drug_W8_new_processing"
elif well == "Well_3":
    # -------- WELL 3 -----------------------:
    adata = sc.read("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/gastrosome_processing/SpaceM_processing_new/Feeding_W3/analysis/single_cell_analysis/spatiomolecular_adata.h5ad") # assuming you only have 1 dataset

    # filter out extracell stuff
    intracell_ions = pd.read_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/AB_Gastrosome_FeedingW3_intra_ions_v1.tsv", sep="\t", index_col=0)
    intracell_ions.head()
    # adata = adata[:, adata.var.formula.isin(intracell_ions["name"])].copy()


    data_dir = Path(r"/Users/alberto-mac/EMBL_ATeam/projects/gastrosome")

    proj_dir = "Feeding_W3_new_processing"
else:
    raise ValueError

plots_path = data_dir / proj_dir / "plots"
plots_path.mkdir(parents=True, exist_ok=True)
sc.settings.figdir = plots_path

cond_col = "cell_type"

adata.obs[cond_col] = np.where(adata.obs["max_intensity-Annotations"] > 0., "Gastrosomes", "Other cells")
adata.obs = adata.obs.astype({cond_col: "category"})

# ------------------------------------
# # Try to select only 172 non-gastrosomes:
# other_cells_ids = np.argwhere((adata.obs[cond_col] == "Other cells").values)[:,0]
# chosen_other_cells_ids = np.random.choice(other_cells_ids, size=172, replace=False)
# mask = adata.obs[cond_col] == "Gastrosomes"
# mask[chosen_other_cells_ids] = True
# print(adata.obs.shape)
# adata = adata[mask, :]
# print(adata.obs.shape)
# ------------------------------------

# ------------------------------------
# # Resample fake gastrosomes
# random_choice = np.random.randint(2, size=adata.obs["max_intensity-Annotations"].shape)
# adata.obs[cond_col] = np.where(random_choice, "Gastrosomes", "Other cells")
# adata.obs = adata.obs.astype({cond_col: "category"})
# ------------------------------------


nb_marked_cells = (adata.obs[cond_col] == "Gastrosomes").sum()
total_nb_cells = adata.obs[cond_col].shape[0]
print("Gastrosomes: {}/{} cells".format(nb_marked_cells, total_nb_cells))

adata.var["moleculeNamesStr"] = adata.var.moleculeNames.apply(convert_name)

# Use ion formula and shortened name as an actual label
adata.var["var_names"] = [ion + " " + name for ion, name in zip(adata.var.index, adata.var.moleculeNamesStr)]

# filter out low ion cells
sc.pp.filter_cells(adata, min_genes=5)

# TIC norm
# sc.pp.normalize_total(adata, key_added='tic')

# Alternative norm method by Alyona:
adata.obs["tic"] = adata.X.sum(axis=1)
# adata.X = np.divide(adata.X, np.array(adata.obs["tic"])[:, None])


# --------------------------------
# FILTERING!
# --------------------------------
# Filtering low and high TIC
sns.histplot(adata.obs, x="tic", hue=cond_col)
plt.title("TIC, unfiltred dataset")
plt.show()
plt.savefig(plots_path / ("unfiltered_tic_%s.png"%cond_col), dpi=300)

lower_thresh = np.quantile(adata.obs["tic"], 0.1)
higher_thresh = np.quantile(adata.obs["tic"], 0.9)
print(lower_thresh, higher_thresh)
adata = adata[(adata.obs["tic"] > lower_thresh) & (adata.obs["tic"] < higher_thresh)]

# Filter not abundant ions
adata.var["log_total_intensity"] = np.log(adata.X.sum(axis=0))
adata.var["nonzero"] = np.count_nonzero(adata.X, axis=0)
adata.var["nonzero_ratio"] = np.count_nonzero(adata.X, axis=0) / adata.X.shape[0]
adata.layers["masked"]= np.ma.masked_less(adata.X, 1)
adata.var["median_nonzero_I"] = np.ma.median(adata.layers["masked"], axis=0)

sns.histplot(adata.var, x="nonzero_ratio")
plt.title("nonzero_ratio, unfiltred dataset")
plt.show()
plt.savefig(plots_path / ("unfiltered_nonzero_ratio_%s.png"%cond_col), dpi=300)

thresh = 0.1
adata = adata[:, adata.var["nonzero_ratio"] > thresh]

# Filter ions not in marked cells:
adata_marked = adata[adata.obs[cond_col] == "Gastrosomes"]
adata_marked.var["log_total_intensity_marked"] = np.log(adata_marked.X.sum(axis=0))
adata_marked.var["nonzero_marked"] = np.count_nonzero(adata_marked.X, axis=0)
adata_marked.var["nonzero_ratio_marked"] = np.count_nonzero(adata_marked.X, axis=0) / adata_marked.X.shape[0]

sns.histplot(adata_marked.var, x="nonzero_ratio_marked")
plt.title("nonzero_ratio_marked, unfiltred dataset")
plt.show()
plt.savefig(plots_path / ("unfiltered_nonzero_ratio_marked_%s.png"%cond_col), dpi=300)

thresh = 0.05
adata = adata[:, adata_marked.var["nonzero_ratio_marked"] > thresh]

sns.histplot(adata.obs, x="tic", hue=cond_col)
plt.title("TIC, unfiltred dataset")
plt.show()

# NORMALIZATION:
# TIC norm
sc.pp.normalize_total(adata, target_sum=1., key_added='tic')
# sc.pp.normalize_total(adata, key_added='tic')

# Alternative norm method by Alyona:
# adata.obs["tic"] = adata.X.sum(axis=1)
# adata.X = np.divide(adata.X, np.array(adata.obs["tic"])[:, None])


# --------------------------
# DE analysis:
# --------------------------
sc.tl.rank_genes_groups(adata, cond_col, method='wilcoxon', key_added="wilcoxon", gene_symbols="var_names")
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon", gene_symbols="var_names")

diff_expr_df = sc.get.rank_genes_groups_df(adata, None, key="wilcoxon", gene_symbols="var_names")

diff_expr_df = diff_expr_df.sort_values("pvals_adj", ascending=True)
diff_expr_df.head(5)

selected = volcano_plot(adata, "wilcoxon", plots_path, pval_thresh=0.05, foldch_thresh=2, gene_symbols="var_names")

# # --------------------------
# # Plot distributions:
# # --------------------------
# dist_plots_path = plots_path / "intensity_distributions"
# dist_plots_path.mkdir(parents=True, exist_ok=True)
# plot_distributions(adata, cond_col, dist_plots_path, gene_symbols="var_names")

# # --------------------------
# # Plot distributions for selected:
# # --------------------------

adata_filtered = adata.copy()
adata_filtered = adata_filtered[:, np.isin(adata_filtered.var["annotation_id"], selected)]
dist_plots_path = plots_path / "intensity_distributions_volcano_selected"
dist_plots_path.mkdir(parents=True, exist_ok=True)
plot_distributions(adata_filtered, cond_col, dist_plots_path, gene_symbols="var_names")
