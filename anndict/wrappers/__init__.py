"""
This module contains adata_dict_fapply* wrappers for packages.
"""
from .anndata_ import (
    set_var_index,
    set_obs_index,
    remove_genes,
    remove_genes_adata_dict,
    add_label_to_adata,
    add_col_to_adata_obs,
    add_col_to_adata_var,
    convert_obs_col_to_category,
    convert_obs_col_to_string,
    convert_obs_index_to_str,
    get_adata_columns,

)

from .scanpy_ import (
    subsample_adata_dict,
    resample_adata,
    resample_adata_dict,
    normalize_adata_dict,
    log_transform_adata_dict,
    set_high_variance_genes_adata_dict,
    rank_genes_groups_adata_dict,
    scale_adata_dict,
    pca_adata_dict,
    neighbors_adata_dict,
    leiden_adata_dict,
    leiden_sub_cluster,
    leiden_sub_cluster_adata_dict,
    calculate_umap_adata_dict,
    plot_umap_adata_dict,

)

# from .squidpy_ import(
#     compute_spatial_neighbors_adata_dict,
#     perform_colocalization_adata_dict,
#     plot_colocalization_adata_dict,
#     compute_interaction_matrix_adata_dict,
#     plot_interaction_matrix_adata_dict,

# )

from .uce_ import (
    UCE_adata,

)

# from .anndictionary_ import (

# )

__all__ = [
    # anndata_
    "set_var_index",
    "set_obs_index",
    "remove_genes",
    "remove_genes_adata_dict",
    "add_label_to_adata",
    "add_col_to_adata_obs",
    "add_col_to_adata_var",
    "convert_obs_col_to_category",
    "convert_obs_col_to_string",
    "convert_obs_index_to_str",
    "get_adata_columns",

    # scanpy_
    "subsample_adata_dict",
    "resample_adata",
    "resample_adata_dict",
    "normalize_adata_dict",
    "log_transform_adata_dict",
    "set_high_variance_genes_adata_dict",
    "rank_genes_groups_adata_dict",
    "scale_adata_dict",
    "pca_adata_dict",
    "neighbors_adata_dict",
    "leiden_adata_dict",
    "leiden_sub_cluster",
    "leiden_sub_cluster_adata_dict",
    "calculate_umap_adata_dict",
    "plot_umap_adata_dict",

    # squidpy_
    # "compute_spatial_neighbors_adata_dict",
    # "perform_colocalization_adata_dict",
    # "plot_colocalization_adata_dict",
    # "compute_interaction_matrix_adata_dict",
    # "plot_interaction_matrix_adata_dict",

    # uce_
    "UCE_adata",
]
