"""
This module contains adata_dict wrappers for `squidpy`.
"""
import squidpy as sq

def compute_spatial_neighbors_adata_dict(adata_dict):
    """
    Computes spatial neighborhood graphs for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    """
    for stratum, adata in adata_dict.items():
        if 'spatial' in adata.obsm:
            # sq.gr.spatial_neighbors(adata, n_neighs=10)
            sq.gr.spatial_neighbors(adata)
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before computing neighbors.")


def perform_colocalization_adata_dict(adata_dict, cluster_key="cell_type"):
    """
    Performs colocalization analysis for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    cluster_key (str): The key in adata.obs containing the cell type or cluster information.
    """
    for stratum, adata in adata_dict.items():
        if 'spatial' in adata.obsm:
            sq.gr.co_occurrence(adata, cluster_key=cluster_key)
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before performing colocalization analysis.")


def plot_colocalization_adata_dict(adata_dict, cluster_key="cell_type", source_cell_type=None, figsize = (10,5)):
    """
    Plots colocalization results for each AnnData object in adata_dict, optionally focusing on a specific source cell type.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    cluster_key (str): The key in adata.obs containing the cell type or cluster information.
    source_cell_type (str, optional): The specific source cell type to focus on in the colocalization plot.
    """
    for stratum, adata in adata_dict.items():
        if 'spatial' in adata.obsm:
            if source_cell_type:
                # Get matches for the source cell type in the cluster key
                matches = [ct for ct in adata.obs[cluster_key].unique() if source_cell_type in ct]
                sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=matches, figsize=figsize)
            else:
                sq.pl.co_occurrence(adata, cluster_key=cluster_key, figsize=figsize)
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before plotting colocalization results.")


def compute_interaction_matrix_adata_dict(adata_dict, cluster_key="cell_type"):
    """
    Computes interaction matrices for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    cluster_key (str): The key in adata.obs containing the cell type or cluster information.
    """
    interaction_matrices = {}
    for stratum, adata in adata_dict.items():
        if 'spatial' in adata.obsm:
            interaction_matrix = sq.gr.interaction_matrix(adata, cluster_key=cluster_key, normalized=True)
            interaction_matrices[stratum] = interaction_matrix
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before computing interaction matrix.")
    return interaction_matrices

def plot_interaction_matrix_adata_dict(adata_dict, cluster_key="cell_type"):
    """
    Plots interaction matrices for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    cluster_key (str): The key in adata.obs containing the cell type or cluster information.
    """
    for stratum, adata in adata_dict.items():
        print(stratum)
        if 'spatial' in adata.obsm:
            sq.pl.interaction_matrix(adata, cluster_key=cluster_key)
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before plotting interaction matrix.")
