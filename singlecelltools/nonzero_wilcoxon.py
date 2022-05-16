from scipy import stats
import numpy as np
import pandas as pd
from typing import Literal, Union, Iterable, Optional
from statsmodels.stats.multitest import multipletests


def nonzero_wilcoxon(
    adata, 
    groupby: str, 
    groups: Union[Literal['all'], Iterable[str]] = 'all', 
    reference: str = "rest", 
    use_raw: bool = True, 
    key_added: Optional[str] = "rank_genes_groups",
    corr_method: str = "benjamini-hochberg", 
    layer: Optional[str] = None,
    copy: bool = False
):

    """\
    Rank features for characterizing groups using a wilcoxon rank-sum test.
    Other than scanpy's `rank_genes_groups`, for each feature only the cells will 
    be used for testing that have a non-zero value.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    use_raw
        Use `raw` attribute of `adata` if present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    corr_method
        p-value correction method.
    key_added
        The key in `adata.uns` information is saved to.
    copy
        Whether to return a copy or operate in-place.

    Returns
    -------
    **names** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    **scores** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    **logfoldchanges** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group. Ordered according to
        scores. Only provided if method is 't-test' like.
        Note: this is an approximation calculated from mean-log values.
    **pvals** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        p-values.
    **pvals_adj** : structured `np.ndarray` (`.uns['rank_genes_groups']`)
        Corrected p-values.
    **pts** : `pandas.DataFrame` (`.uns['rank_genes_groups']`)
        Fraction of cells expressing the genes for each group.
    **pts_rest** : `pandas.DataFrame` (`.uns['rank_genes_groups']`)
        Fraction of cells from the union of the rest of each group
        expressing the genes.
    """

    params = {
        "groupby": groupby,
        "reference": reference,
        "method": "wilcoxon",
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method
    }

    if copy:
        adata = adata.copy()

    if adata.obs[groupby].dtype != "category":
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    if groups == "all":
        groupby_groups = adata.obs[groupby].cat.categories
    else:
        groupby_groups = groups

    if use_raw:
        X = adata.raw.X.copy()
    elif layer:
        X = adata.layers[layer].copy()
    else:
        X = adata.X.copy()

    # If data has been 
    if "log1p" in adata.uns_keys():
        X = np.expm1(X)

    # Mask zero counts by setting them to nan
    X[X == 0] = np.nan

    n_genes = X.shape[1]
    scores = np.zeros(n_genes)
    T = 1 # No tie correction

    test_stat_columns = ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]
    test_stats = {key: pd.DataFrame() for key in test_stat_columns}

    pts = pd.DataFrame()
    pts_rest = pd.DataFrame()

    # Do wilcoxon test for each group
    for group in groupby_groups:

        mask = adata.obs[groupby] == group
        if reference == "rest":
            mask_ref = ~mask
        else:
            mask_ref = adata.obs[groupby] == reference

        # Some basic stats used for various metrics
        n_means = np.nanmean(X[mask], axis=0)
        m_means = np.nanmean(X[mask_ref], axis=0)

        n_active = np.count_nonzero(mask)
        n_nz_active = np.sum(~np.isnan(X[mask]), axis=0, dtype="i8")

        m_active = np.count_nonzero(mask_ref)
        m_nz_active = np.sum(~np.isnan(X[mask_ref]), axis=0, dtype="i8")

        ranks = pd.DataFrame(data=np.vstack([X[mask], X[mask_ref]]))
        ranks = ranks.rank()

        scores = np.sum(ranks.iloc[0:n_active, :])

        std_dev = np.sqrt(
            T * n_nz_active * m_nz_active * (n_nz_active + m_nz_active + 1) / 12.0
        )

        # Compute wilcoxon scores, z-transform them
        scores = (
            scores - (n_nz_active * ((n_nz_active + m_nz_active + 1) / 2.0))
        ) / std_dev

        scores[np.isnan(scores)] = 0

        # Compute pvals from wilcoxon scores
        foldchanges = (n_means + 1e-9) / (m_means + 1e-9)
        logfoldchanges = np.log2(foldchanges)

        pvals = 2 * stats.distributions.norm.sf(np.abs(scores))
        
        # Pval correction
        if corr_method == 'benjamini-hochberg':

            pvals[np.isnan(pvals)] = 1
            _, pvals_adj, _, _ = multipletests(
                pvals, alpha=0.05, method='fdr_bh'
            )
        elif corr_method == 'bonferroni':
            pvals_adj = np.minimum(pvals * n_genes, 1.0)
        else:
            raise ValueError(f"Unknown correction method: {corr_method}")

        scores_df = pd.DataFrame(
            {
                "names": adata.var.index,
                "scores": scores,
                "logfoldchanges": logfoldchanges,
                "pvals": pvals,
                "pvals_adj": pvals_adj
            }
        )

        scores_df.sort_values("scores", ascending=False, inplace=True)
        scores_df.reset_index(inplace=True, drop=True)

        for col in scores_df:
            test_stats[col][group] = scores_df[col]

        pts[group]      = pd.Series(n_nz_active / n_active, index=adata.var.index)
        pts_rest[group] = pd.Series(m_nz_active / m_active, index=adata.var.index)

    dtypes = {
        'names': 'O',
        'scores': 'float32',
        'logfoldchanges': 'float32',
        'pvals': 'float64',
        'pvals_adj': 'float64'
    }
    
    adata.uns[key_added] = {
        "params": params,
        "pts": pts,
        "pts_rest": pts_rest
    }

    # transfer stats into adata.uns
    for col in test_stats:
        adata.uns[key_added][col] = test_stats[col].to_records(index=False, column_dtypes=dtypes[col])

    if copy:
        return adata
