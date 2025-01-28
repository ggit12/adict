"""
This module contains adata_dict wrappers for `scanpy`.
"""

import scanpy as sc

from anndict.adata_dict import adata_dict_fapply, adata_dict_fapply_return
from anndict.adata_dict import build_adata_dict, concatenate_adata_dict, check_and_create_stratifier


def subsample_adata_dict(adata_dict, **kwargs):
    """
    Subsamples each AnnData object in the dictionary using Scanpy's subsample function.
    
    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the subsample function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    n_obs = kwargs.get('n_obs', None)
    fraction = kwargs.get('fraction', None)

    if n_obs is None and fraction is None:
        fraction = 1
        kwargs['fraction'] = fraction

    def subsample_adata(adata, **kwargs):
        if n_obs is None or adata.n_obs > n_obs:
            sc.pp.subsample(adata, **kwargs)

    adata_dict_fapply(adata_dict, subsample_adata, **kwargs)


def resample_adata(adata, strata_keys, min_num_cells, n_largest_groups=None, **kwargs):
    """
    Resample an AnnData object based on specified strata keys and drop strata with fewer than the minimum number of cells.

    Parameters:
    adata (AnnData): Annotated data matrix.
    strata_keys (list of str): List of column names in adata.obs to use for stratification.
    min_num_cells (int): Minimum number of cells required to retain a stratum.
    kwargs: Additional keyword arguments to pass to the subsample function.

    Returns:
    AnnData: Concatenated AnnData object after resampling and filtering.

    Raises:
    ValueError: If any of the specified strata_keys do not exist in adata.obs.
    """
    # Step 1: Create the strata key
    strata_key = check_and_create_stratifier(adata, strata_keys)

    # Step 2: Calculate the size of each category
    category_counts = adata.obs[strata_key].value_counts()

    # Step 3: Identify the top n largest categories or all categories if n is None
    if n_largest_groups is None:
        selected_categories = category_counts.index.tolist()
    else:
        selected_categories = category_counts.nlargest(n_largest_groups).index.tolist()

    # Step 4: Build adata_dict based on the strata key
    strata_dict = build_adata_dict(adata, [strata_key], desired_strata=selected_categories)

    # Step 5: Subsample each AnnData object in the strata_dict
    subsample_adata_dict(strata_dict, **kwargs)

    # Step 6: Drop AnnData objects with fewer than min_num_cells
    filtered_dict = {k: v for k, v in strata_dict.items() if v.n_obs >= min_num_cells}

    # Step 7: Concatenate the filtered_dict back to a single AnnData object
    #setting index_unique=None avoids index modification
    return concatenate_adata_dict(filtered_dict, index_unique=None)


def resample_adata_dict(adata_dict, strata_keys, n_largest_groups=None, min_num_cells=0, **kwargs):
    """
    Resample each AnnData object in a dictionary based on specified strata keys and drop strata with fewer than the minimum number of cells.

    Parameters:
    adata_dict (dict): Dictionary where keys are strata values and values are AnnData objects.
    strata_keys (list of str): List of column names in adata.obs to use for stratification.
    min_num_cells (int, optional): Minimum number of cells required to retain a stratum. Default is 0.
    kwargs: Additional keyword arguments to pass to the resample function.

    Returns:
    dict: Dictionary of resampled AnnData objects after filtering.
    """
    return adata_dict_fapply_return(adata_dict, resample_adata, strata_keys=strata_keys, n_largest_groups=n_largest_groups, min_num_cells=min_num_cells, **kwargs)


def normalize_adata_dict(adata_dict, **kwargs):
    """
    Normalizes each AnnData object in the dictionary using Scanpy's normalize_total.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the normalize_total function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.normalize_total, **kwargs)


def log_transform_adata_dict(adata_dict, **kwargs):
    """
    Log-transforms each AnnData object in the dictionary using Scanpy's log1p.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the log1p function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.log1p, **kwargs)


def set_high_variance_genes_adata_dict(adata_dict, **kwargs):
    """
    Identifies high-variance genes in each AnnData object in the dictionary.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the highly_variable_genes function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.highly_variable_genes, **kwargs)

def rank_genes_groups_adata_dict(adata_dict, **kwargs):
    """
    Identifies differentially expressed genes in each AnnData object in the dictionary.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the rank_genes_groups function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.tl.rank_genes_groups, **kwargs)


def scale_adata_dict(adata_dict, **kwargs):
    """
    Scales each AnnData object in the dictionary using Scanpy's scale function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the scale function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.scale, **kwargs)


def pca_adata_dict(adata_dict, **kwargs):
    """
    Performs PCA on each AnnData object in the dictionary using Scanpy's pca function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the pca function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.pca, **kwargs)


def neighbors_adata_dict(adata_dict, **kwargs):
    """
    Calculates neighborhood graph for each AnnData object in the dictionary using Scanpy's neighbors function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the sc.pp.neighbors function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.neighbors, **kwargs)


def leiden_adata_dict(adata_dict, **kwargs):
    """
    Performs Leiden clustering for each AnnData object in the dictionary using Scanpy's leiden function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the sc.tl.leiden function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.tl.leiden, **kwargs)

def leiden_sub_cluster(adata, groupby, **kwargs):
    """
    Perform Leiden clustering on subgroups of cells.
    This function applies Leiden clustering to subgroups of cells defined by the groupby parameter.

    Parameters:
    adata : AnnData Annotated data matrix.
    groupby : str Column name in adata.obs for grouping cells before subclustering.
    kwargs : dict Additional keyword arguments to pass to the leiden_adata_dict function.

    Returns:
    None, The function modifies the input AnnData object in-place.
    """
    adata_dict = build_adata_dict(adata, strata_keys=[groupby])
    leiden_adata_dict(adata_dict, **kwargs)
    adata = concatenate_adata_dict(adata_dict, index_unique=None) #setting index_unique=None avoids index modification
    return adata


def leiden_sub_cluster_adata_dict(adata_dict, groupby, **kwargs):
    """
    This function applies the leiden_sub_cluster function to each AnnData object
    in the provided dictionary.
    
    Parameters:
    adata_dict : dict Dictionary of AnnData objects.
    groupby : str Column name in adata.obs for grouping cells before subclustering.
    kwargs : dict Additional keyword arguments to pass to the leiden_sub_cluster function.

    Returns:
    None The function modifies the input AnnData objects in-place.
    """
    return adata_dict_fapply_return(adata_dict, leiden_sub_cluster, groupby=groupby, **kwargs)


def calculate_umap_adata_dict(adata_dict, **kwargs):
    """
    Calculates UMAP embeddings for each subset in the adata_dict.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'use_rep' which specifies the key in .obsm where the representation matrix is stored.

    Returns:
    - dict: A dictionary with the same keys as adata_dict, but values now include UMAP coordinates.
    """
    adata_dict_fapply(adata_dict, sc.tl.umap, **kwargs)
    return adata_dict


def plot_umap_adata_dict(adata_dict, **kwargs):
    """
    Plots UMAP embeddings for each AnnData object in adata_dict, colored by a specified variable.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'color_by' which specifies a variable by which to color the UMAP plots, typically a column in .obs.

    Returns:
    - None: The function creates plots for the AnnData objects.
    """
    def plot_umap(adata, adt_key=None, **kwargs):
        print(f"Plotting UMAP for key: {adt_key}")
        if 'X_umap' in adata.obsm:
            sc.pl.umap(adata, **kwargs)
        else:
            print(f"UMAP not computed for adata with key {adt_key}. Please compute UMAP before plotting.")
    adata_dict_fapply(adata_dict, plot_umap, use_multithreading=False, **kwargs)
