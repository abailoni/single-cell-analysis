import os.path

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

import scanpy as sc
import pandas as pd
import seaborn as sns
# from outer_spacem.io import convert_name

import numpy as np
from pathlib import Path

# from outer_spacem.pl import plot_distributions, plot_umap_top_n, volcano_plot
from outer_spacem.pl._diff_expr import plot_distributions
# from singlecelltools.various import get_molecules_names

well = "Drug_W8"
analysis_name = "magic_v1.1"

main_dir = os.path.join("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome",
                       well, "reprocessing")

adata = sc.read(os.path.join(main_dir, "single_cell_analysis/spatiomolecular_adata.h5ad"))

# intracell_ions = pd.read_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/molecules_databases/reannotated/AB_Gastrosome_DrugW8_intra_ions_v2.tsv",
                             # sep="\t", index_col=0)

proj_dir = os.path.join(main_dir, "analysis", analysis_name)
plots_path = os.path.join(proj_dir, "plots")

os.makedirs(plots_path, exist_ok=True)
sc.settings.figdir = plots_path


cond_col = "Cell type"
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


# # ---------------------------
# # Fix the mess with the molecules names:
# # ---------------------------
#
# # Get original databases properly annotated:
# CM = pd.read_csv(
#     "/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/molecules_databases/core_metabolome_v3.csv",
#     sep="\t", index_col=0)
#
# SwissLip = pd.read_csv(
#     "/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/molecules_databases/swisslipids_2018-02-02-v2.tsv",
#     sep="\t", index_col=0)
#
# CM.to_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/Drug_W8/test_data.csv")
#
# combined_databases = pd.concat([CM, SwissLip])
# merged_database = combined_databases.merge(adata.var, on='formula')


print("Cells before filtering:", adata.shape[0])
sc.pp.filter_cells(adata, min_genes=10)
print("Cells after filtering:", adata.shape[0])

print("Ions before filtering:", adata.shape[1])
sc.pp.filter_genes(adata, min_cells=200) # 200
print("Ions after filtering:", adata.shape[1])


# Try out MAGIC:
sc.pp.log1p(adata)
sc.pp.normalize_total(adata, key_added='tic')
adata_pre_magic = adata.copy()
sc.external.pp.magic(adata, name_list="all_genes", t=5, solver="exact")

old_stuff = False

if old_stuff:
    # Get rid of all molecules that have any zero ion...?!
    # zero_mask = adata.X == 0
    adata.obs["nonzero_ratio_ions"] = np.count_nonzero(adata.X, axis=1) / adata.X.shape[1]
    sns.histplot(adata.obs, x="nonzero_ratio_ions")
    plt.title("Ratio nonzero ions in cell")
    plt.ylabel("Cell count")
    plt.xlabel("Ratio nonzero ions in cell")
    plt.savefig(os.path.join(plots_path, ("nonzero_ratio_ions.png")), dpi=300)
    plt.show()

    adata.obs["tic"] = adata.X.sum(axis=1)

    # --------------------------------
    # FILTERING!
    # --------------------------------
    # Filtering low and high TIC
    sns.histplot(adata.obs, x="tic", hue=cond_col)
    plt.title("TIC, unfiltred dataset")
    plt.savefig(os.path.join(plots_path, ("unfiltered_tic_%s.png"%cond_col)), dpi=300)
    plt.show()

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
    plt.savefig(os.path.join(plots_path, "unfiltered_nonzero_ratio_%s.png"%cond_col), dpi=300)
    plt.show()

    thresh = 0.1
    adata = adata[:, adata.var["nonzero_ratio"] > thresh]

    # Filter ions not in marked cells:
    adata_marked = adata[adata.obs[cond_col] == "Gastrosomes"]
    adata_marked.var["log_total_intensity_marked"] = np.log(adata_marked.X.sum(axis=0))
    adata_marked.var["nonzero_marked"] = np.count_nonzero(adata_marked.X, axis=0)
    adata_marked.var["nonzero_ratio_marked"] = np.count_nonzero(adata_marked.X, axis=0) / adata_marked.X.shape[0]

    sns.histplot(adata_marked.var, x="nonzero_ratio_marked")
    plt.title("nonzero_ratio_marked, unfiltred dataset")
    plt.savefig(os.path.join(plots_path, ("unfiltered_nonzero_ratio_marked_%s.png"%cond_col)), dpi=300)
    plt.show()

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
# adata.X = np.log1p(adata.X)
sc.tl.rank_genes_groups(adata,
                        cond_col,
                        method='wilcoxon',
                        key_added="rank_genes_groups")
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon", gene_symbols="var_names")

groupname = adata.uns["rank_genes_groups"]["params"]["groupby"]  # = "leiden"

pval_thres = 0.05  # upper threshold for p-values
fc_thres = 2  # lower threshold for fold changes

for group in adata.obs[groupname].unique().categories:
    if group == "Gastrosomes":
        # df = sc.get.rank_genes_groups_df(adata, None, key="rank_genes_groups", gene_symbols="moleculesNames")
        df = sc.get.rank_genes_groups_df(adata, group, )
        df = df.sort_values("scores", ascending=False)

        df.insert(0, groupname, group)

        significance = (df["pvals_adj"] < pval_thres) & (df["logfoldchanges"].abs() > np.log2(fc_thres))
        df["significance"] = np.where(significance, "Other cells", "-")
        df["significance"][significance & (df["logfoldchanges"] > 0)] = "Gastrosomes"
        df["significance"].astype("category")

        df["pvals_adj_nlog10"] = -np.log10(df["pvals_adj"] + np.finfo("f8").eps)

        # Create an array with the colors you want to use
        colors = [
            "#d62728", #Red
            # "#e377c2", #Pink
                  "#C7C7C7", #
            "#1f77b4", # Blue
                  "#17becf"]

        # Set your custom color palette
        sns.set_palette(sns.color_palette(colors))

        plt.figure(figsize=[15, 5])
        sns.scatterplot(
            data=df,
            x="logfoldchanges",
            y="pvals_adj_nlog10",
            s=10,
            linewidth=0,
            hue="significance",
            # palette="tab10"
            legend=False
        )
        plt.xlabel("Fold Changes: $\log_2 (FC)$")
        plt.ylabel("p-values: $-\log_{10}(p)$")
        plt.legend(loc="lower left", title="Significance",
                   labels=['Gastrosomes (up)',
                           '-',
                           "Other cells (down)"])
        plt.title(f"Cell type: Gastrosomes", fontsize=20)
        # plt.title(f"{groupname}={group}", fontsize=20)
        plt.savefig(os.path.join(plots_path, f"volcano_plot_{group}.png"), dpi=300)
        plt.show()

        df = df.drop(columns="Cell type")

        # Add molecules names:
        df = df.rename(columns={'names': 'annotation_id',
                                "significance": "Cell Type"})
        df = pd.merge(df, adata.var[['annotation_id', 'moleculeNames']],
                      on="annotation_id",
                      how='left')

        # Save relevant scores in file:
        selected_df = df[significance]

        df_path = "{}/{}_{}_markers.csv".format(plots_path, groupname.replace(" ", "_"), group)

        print(df_path)
        selected_df.to_csv(df_path, index=False)

        # Export distributions:
        for adata_to_plot in [(adata, "post_magic"), (adata_pre_magic, "pre_magic")]:
            for i, export_group_name in enumerate(["Gastrosomes", "Other cells"]):
                adata_filtered = adata_to_plot[0].copy()

                # adata_filtered.var.astype({'annotation_id': 'string'}, copy=False)
                # adata_filtered.var = adata_filtered.var.set_index('annotation_id')
                # adata_filtered.var = adata_filtered.var.reindex(index=df['annotation_id'])
                # adata_filtered.var = adata_filtered.var.reset_index()

                # df.astype({'annotation_id': 'string'}, copy=False)
                merged_df = pd.merge(adata_filtered.var,
                                          df[['annotation_id', 'Cell Type']],
                              on="annotation_id",
                              how='left')
                # adata_filtered.var.astype({'Cell Type': 'category'}, copy=False)
                adata_filtered = adata_filtered[:, merged_df["Cell Type"] == export_group_name]
                # adata_filtered = adata_filtered[:, adata_filtered.var["annotation_id"] == "C4H9O7P+Na"]
                dist_plots_path = os.path.join(plots_path, "distributions_{}".format(adata_to_plot[1]), export_group_name.replace(" ", "_", ))
                os.makedirs(dist_plots_path, exist_ok=True)
                plot_distributions(adata_filtered, cond_col, Path(dist_plots_path), gene_symbols="annotation_id")



# selected = volcano_plot(adata, "wilcoxon", plots_path, pval_thresh=0.05, foldch_thresh=2, gene_symbols="var_names")

# # Export results to csv:
# diff_expr_df = sc.get.rank_genes_groups_df(adata, None, key="wilcoxon", gene_symbols="var_names")
# diff_expr_df = diff_expr_df.sort_values("pvals_adj", ascending=True)
# diff_expr_df = diff_expr_df[diff_expr_df["group"] == "Gastrosomes"]
# diff_expr_df.to_csv(os.path.join(plots_path, "DE_results.csv"))

# # --------------------------
# # Plot distributions:
# # --------------------------
# dist_plots_path = plots_path / "intensity_distributions"
# dist_plots_path.mkdir(parents=True, exist_ok=True)
# plot_distributions(adata, cond_col, dist_plots_path, gene_symbols="var_names")

# # --------------------------
# # Plot distributions for selected:
# # --------------------------

