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
# from outer_spacem.pl._diff_expr import plot_distributions
# from singlecelltools.various import get_molecules_names

SAVE_ADATA = False
MAGIC = False

well = "Drug_W8"
export_only_significant = True
#
# well = "Feeding_W3"
# export_only_significant = False

analysis_name = "v4"
if MAGIC: analysis_name += "_magic"

main_dir = os.path.join("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome",
                       well, "reprocessing")

# adata = sc.read(os.path.join(main_dir, "single_cell_analysis/spatiomolecular_adata.h5ad"))
# adata = sc.read("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/gastrosome_processing/SpaceM_processing_new/Drug_W8/analysis/single_cell_analysis/spatiomolecular_adata.h5ad")

# # //////////////////
# # New import where I import another well with fixed molecule IDs:
# metadata = pd.read_csv("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/gastrosome_processing_full/spacem/spacem_datasets_paths_final.csv")
# # metadata_subset = metadata.loc[metadata.index.isin([10]), :]
# # metadata_subset = metadata.loc[metadata.index.isin([3, 10]), :]
# metadata_subset = metadata
# # metadata_subset = metadata.loc[metadata.index.isin([0, 3]), :]
# import warnings
# import outer_spacem as osm
# pattern = "/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/projects/gastrosome_processing_full/spacem_synced/slide{slide}/W{well}/analysis/single_cell_analysis/spatiomolecular_adata.h5ad"
#
# warnings.simplefilter("ignore")
# adata = osm.io.bulk_read(
#     metadata_subset,
#     file_pattern = pattern
# )
# # Keep only drug treated cells:
# adata = adata[adata.obs.condition == "Drugs"]
# # mol_ids_col = "moleculeIds-0"
# # mol_names_col = "moleculeNames-0"

# //////////////////
# Old direct import:
adata = sc.read("/Users/alberto-mac/Documents/DA_ESPORTARE/LOCAL_EMBL_FILES/scratch/bailoni/projects/gastrosome_processing_full/spacem_synced/slide6/W8/analysis/single_cell_analysis/spatiomolecular_adata.h5ad")
#
# # metadata_subset = metadata.loc[metadata.index.isin([3]), :]
# # adata2 = osm.io.bulk_read(
# #     metadata_subset,
# #     file_pattern = pattern
# # )
mol_ids_col = "moleculeIds"
mol_names_col = "moleculeNames"

# //////////////////


# intracell_ions = pd.read_csv("/Users/alberto-mac/EMBL_ATeam/projects/gastrosome/molecules_databases/reannotated/AB_Gastrosome_DrugW8_intra_ions_v2.tsv",
                             # sep="\t", index_col=0)

proj_dir = os.path.join(main_dir, "analysis", analysis_name)

plots_path = os.path.join(proj_dir, "plots")

os.makedirs(plots_path, exist_ok=True)
sc.settings.figdir = plots_path


cond_col = "Cell type"
# adata.obs[cond_col] = np.where(adata.obs["max_intensity-Annotations"] > 0., "Gastrosomes", "Other cells treated with drug")
adata.obs[cond_col] = np.where(adata.obs["max_intensity-Gastrosomes"] > 0., "Gastrosomes", "Other cells treated with drug")
adata.obs = adata.obs.astype({cond_col: "category"})

# ------------------------------------
# # Try to select only 172 non-gastrosomes:
# other_cells_ids = np.argwhere((adata.obs[cond_col] == "Other cells treated with drug").values)[:,0]
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
# adata.obs[cond_col] = np.where(random_choice, "Gastrosomes", "Other cells treated with drug")
# adata.obs = adata.obs.astype({cond_col: "category"})
# ------------------------------------


nb_marked_cells = (adata.obs[cond_col] == "Gastrosomes").sum()
total_nb_cells = adata.obs[cond_col].shape[0]
print("Gastrosomes: {}/{} cells".format(nb_marked_cells, total_nb_cells))


# Get INCHI names:
from singlecelltools.various import get_inchi_and_ids
if not SAVE_ADATA:
    adata = get_inchi_and_ids(adata, copy=False, name_mol_names_col=mol_names_col,
                              out_name_molId_col="moleculeIds")

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
if MAGIC:
    adata = sc.external.pp.magic(adata, name_list="all_genes", t=5, solver="exact", copy=True)

if SAVE_ADATA:
    # Save modified adata to file:
    # adata.obs.rename(columns={"Cell type": "cell_type"})
    adata.write(Path(plots_path) / "adata_filtered.h5ad")
    if MAGIC:
        adata_pre_magic.write(Path(plots_path) / "adata_filtered_pre_magic.h5ad")
    exit(0)

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
        df["Significance"] = np.where(significance, "True (Gastrosomes down)", "False")
        df["Significance"][significance & (df["logfoldchanges"] > 0)] = "True (Gastrosomes up)"
        df["Significance"].astype("category")

        df["pvals_adj_nlog10"] = -np.log10(df["pvals_adj"] + np.finfo("f8").eps)

        df["pvals_adj_bonferroni"] = df["pvals_adj"] / adata.X.shape[0]

        # Create an array with the colors you want to use
        colors = [
            "#d62728", #Red
            # "#e377c2", #Pink
                  "#C7C7C7", #
            "#1f77b4", # Blue
                  "#17becf"]

        # Set your custom color palette
        sns.set(font_scale=1.2)
        sns.set_palette(sns.color_palette(colors))



        df = df.drop(columns="Cell type")

        # Add molecules names:
        df = df.rename(columns={'names': 'annotation_id',
                                })
        df = pd.merge(df, adata.var[['annotation_id', mol_ids_col, mol_names_col]],
                      on="annotation_id",
                      how='left')

        df = df.rename(columns={
            mol_names_col: 'moleculeNames',
            mol_ids_col: 'moleculeIds', })



        plt.figure(figsize=[15, 5])

        sns.scatterplot(
            data=df,
            x="logfoldchanges",
            y="pvals_adj_nlog10",
            s=10,
            linewidth=0,
            hue="Significance",
            # size=15,
            # palette="tab10"
            # legend=False,
        )
        plt.xlabel("Fold Changes: $\log_2 (FC)$")
        plt.ylabel("p-values: $-\log_{10}(p)$")
        # plt.legend(loc="lower left", title="Significance",
        #            labels=['Gastrosomes (up)',
        #                    '-',
        #                    "Other cells treated with drug (down)"])
        # plt.legend(loc="lower left", title="Significance",
        #            # labels=['Gastrosomes (up)',
        #            #         '-',
        #            #         "Other cells treated with drug (down)"]
        #            )
        plt.title(f"Gastrosomes vs other cells treated with drugs", fontsize=20)
        # plt.title(f"{groupname}={group}", fontsize=20)
        plt.xlim(-3,3)

        # # Add marker labels:
        # line_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # line_colors *= 100
        # # for (row_i, col_j, cue), data_ijk in g.facet_data():
        # #     ax = g.facet_axis(row_i, col_j)
        # # batch_name = data_ijk[BATCH_KEY_COL].unique()[0]
        # # covered_markers = covered_markers_collected[batch_name]
        # markers_to_label = df[df["Significance"] != "False"]
        #
        # # Get plot limits and place labels randomly:
        # xlims = plt.xlim()
        # ylims = plt.ylim()
        # x_range = xlims[1] - xlims[0]
        # y_range = ylims[1] - ylims[0]
        #
        # for i, (_, row) in enumerate(markers_to_label.iterrows()):
        #     text_x = row["logfoldchanges"] + np.random.rand() * 0.05
        #     text_y = row["pvals_adj_nlog10"] + (np.random.rand()) * 0.05 * df["pvals_adj_nlog10"].max()
        #     # text_x = xlims[0] + x_range * 0.1 + (np.random.rand() * (x_range * 0.6))
        #     # text_y = ylims[0] + y_range * 0.1 + (np.random.rand() * (y_range * 0.8))
        #     plt.plot((row["logfoldchanges"], text_x), (row["pvals_adj_nlog10"], text_y), alpha=0.6,
        #              color=line_colors[i])
        #
        #     # Get the first name:
        #     # plt.text(text_x, text_y, s=f"{row['ion']} ({eval(row['moleculeNames'])[0]})", alpha=0.9)
        #     plt.text(text_x, text_y, s=f"{row['ion']}", alpha=0.9, size=10)
        #     # break

        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"volcano_plot_{group}.pdf"), dpi=300)
        plt.show()

        df.rename(columns={"annotation_id": "ion"}, inplace=True)
        reordered_cols = ['Significance', 'ion', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj',
                          'pvals_adj_bonferroni',
                          'moleculeIds',
                          'moleculeNames']
        df = df[reordered_cols]

        # Save relevant scores in file:
        if export_only_significant:
            selected_df = df[significance]
        else:
            selected_df = df


        df_path = "{}/{}_{}_markers.csv".format(plots_path, groupname.replace(" ", "_"), group)

        print(df_path)
        # selected_df["Cell Type"] = np.where(selected_df["Cell Type"] == group, group + " (up)",
        #                                   group + " (down)")
        selected_df.to_csv(df_path, index=False)

        # # Export distributions:
        # for adata_to_plot in [(adata, "post_magic"), (adata_pre_magic, "pre_magic")]:
        #     for i, export_group_name in enumerate(["Gastrosomes", "Other cells treated with drug"]):
        #         adata_filtered = adata_to_plot[0].copy()
        #
        #         # adata_filtered.var.astype({'annotation_id': 'string'}, copy=False)
        #         # adata_filtered.var = adata_filtered.var.set_index('annotation_id')
        #         # adata_filtered.var = adata_filtered.var.reindex(index=df['annotation_id'])
        #         # adata_filtered.var = adata_filtered.var.reset_index()
        #
        #         # df.astype({'annotation_id': 'string'}, copy=False)
        #         merged_df = pd.merge(adata_filtered.var,
        #                                   df[['annotation_id', 'Significance']],
        #                       on="annotation_id",
        #                       how='left')
        #         # adata_filtered.var.astype({'Significance': 'category'}, copy=False)
        #         adata_filtered = adata_filtered[:, merged_df["Significance"] == export_group_name]
        #         # adata_filtered = adata_filtered[:, adata_filtered.var["annotation_id"] == "C4H9O7P+Na"]
        #         dist_plots_path = os.path.join(plots_path, "distributions_{}".format(adata_to_plot[1]), export_group_name.replace(" ", "_", ))
        #         os.makedirs(dist_plots_path, exist_ok=True)
                # plot_distributions(adata_filtered, cond_col, Path(dist_plots_path), gene_symbols="annotation_id")



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

