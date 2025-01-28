"""
Label transfer via harmony integration.
"""

import anndata as ad
import scanpy as sc
import harmonypy as hm

def harmony_label_transfer(adata_to_label, master_data, master_subset_column, label_column):
    """
    Perform Harmony integration and transfer labels from master_data to adata_to_label.

    This function subsets master_data based on a provided column to get the cells
    that match in the same column of adata_to_label. It then performs Harmony
    integration on the combined dataset and transfers the specified label column
    from master_data to adata_to_label.

    Parameters:
    adata_to_label : anndata.AnnData The AnnData object to which labels will be transferred.
    master_data : anndata.AnnData The master AnnData object containing the reference data and labels.
    master_subset_column : str The column name in .obs used for subsetting master_data to match adata_to_label.
    label_column : str The column name in .obs of master_data containing the labels to be transferred.

    Returns:
    anndata.AnnData The adata_to_label object with a new column 'harmony_labels' in .obs containing the transferred labels.
    """

    # Subset master_data based on the provided column to get matching cells
    matching_cells = master_data.obs[master_data.obs[master_subset_column].isin(adata_to_label.obs[master_subset_column])]
    master_subset = master_data[matching_cells.index]

    # Combine adata_to_label and the subset of master_data
    combined_data = ad.concat([adata_to_label, master_subset])

    # Perform Harmony integration
    sc.tl.pca(combined_data, svd_solver='arpack')
    harmony_results = hm.run_harmony(combined_data.obsm['X_pca'], combined_data.obs, master_subset_column)
    combined_data.obsm['X_harmony'] = harmony_results.Z_corr.T

    # Separate the integrated data back into the original datasets
    adata_to_label_integrated = combined_data[:adata_to_label.n_obs]
    master_integrated = combined_data[adata_to_label.n_obs:]

    # Transfer labels from master_data to adata_to_label using the integrated data
    sc.pp.neighbors(master_integrated, use_rep='X_harmony')
    sc.tl.umap(master_integrated)
    sc.tl.leiden(master_integrated, resolution=0.5)

    # Transfer the specific label column from master_integrated to adata_to_label_integrated
    master_labels = master_integrated.obs[label_column]
    adata_to_label_integrated.obs[label_column] = master_labels.reindex(adata_to_label_integrated.obs.index).fillna(method='ffill')

    # Return adata_to_label with the new labels
    adata_to_label.obs['harmony_labels'] = adata_to_label_integrated.obs[label_column]
