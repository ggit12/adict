"""
This module contains adata_dict wrappers for functions in `anndict`.
"""


def ai_annotate_biological_process_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_biological_process'):
    """
    Applies ai_annotate_biological_process to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_biological_process, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def simplify_var_index_adata_dict(adata_dict, column, new_column_name, simplification_level=''):
    """
    Applies simplify_var_index to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, simplify_var_index, max_retries=3, column=column, new_column_name=new_column_name, simplification_level=simplification_level)


def ensure_label_consistency_adata_dict(adata_dict, cols, simplification_level='unified, typo-fixed', new_col_prefix='consistent'):
    """
    Apply label consistency across multiple AnnData objects in a dictionary.

    This function applies ensure_label_consistency_adata to each AnnData in adata_dict.
    
    Parameters:
    adata_dict : dict Dictionary of AnnData objects.
    cols : list List of column names in adata.obs for which label consistency is enforced.
    simplification_level : str, optional Level of label simplification (default is 'unified, typo-fixed').
    new_col_prefix : str, optional Prefix for the new consistent label columns (default is 'consistent').

    See ensure_label_consistency_adata for details.
    """
    return adata_dict_fapply_return(adata_dict, ensure_label_consistency_adata, cols=cols, simplification_level=simplification_level, new_col_prefix=new_col_prefix)


def ai_annotate_cell_type_by_comparison_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_cell_type_by_comparison', cell_type_of_origin_col=None, tissue_of_origin_col=None, **kwargs):
    """
    Applies ai_annotate_cell_type_by_comparison to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type_by_comparison, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column, cell_type_of_origin_col=cell_type_of_origin_col, tissue_of_origin_col=tissue_of_origin_col, **kwargs)


def simplify_obs_column_adata_dict(adata_dict, column, new_column_name, simplification_level=''):
    """
    Applies simplify_obs_column to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, simplify_obs_column, max_retries=3, column=column, new_column_name=new_column_name, simplification_level=simplification_level)


def create_label_hierarchy_adata_dict(adata_dict, col, simplification_levels):
    """
    Applies create_label_hierarchy to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, create_label_hierarchy, max_retries=3, col=col, simplification_levels=simplification_levels)


def ai_annotate_cell_type_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_cell_type', tissue_of_origin_col=None):
    """
    Applies ai_annotate_cell_type to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column, tissue_of_origin_col=tissue_of_origin_col)


def ai_compare_cell_type_labels_pairwise_adata_dict(adata_dict, cols1, cols2, new_col_prefix='agreement', comparison_level='binary'):
    """
    Applies ai_compare_cell_type_labels_pairwise to each anndata in an anndict.
    """
    return adata_dict_fapply_return(adata_dict, ai_compare_cell_type_labels_pairwise, max_retries=3, cols1=cols1, cols2=cols2, new_col_prefix=new_col_prefix, comparison_level=comparison_level)


def ai_annotate_cell_sub_type_adata_dict(adata_dict, cell_type_col, sub_cluster_col, new_label_col, tissue_of_origin_col=None, n_top_genes=10):
    """
    Annotate cell subtypes for a dictionary of AnnData objects.

    This function applies the ai_annotate_cell_sub_type function to each AnnData object
    in the provided dictionary.

    Parameters:
    adata_dict : dict Dictionary of AnnData objects.
    cell_type_col : str Column name in adata.obs containing main cell type labels.
    new_label_col : str Name of the column to store the AI-generated subtype labels.

    Returns:
    dict Dictionary of annotated AnnData objects with AI-generated subtype labels.
    """
    results = adata_dict_fapply_return(adata_dict, ai_annotate_cell_sub_type, max_retries=3, cell_type_col=cell_type_col, sub_cluster_col=sub_cluster_col, new_label_col=new_label_col, tissue_of_origin_col=tissue_of_origin_col, n_top_genes=n_top_genes)
    annotated_adata_dict = {key: result[0] for key, result in results.items()}
    label_mappings_dict = {key: result[1] for key, result in results.items()}

    return annotated_adata_dict, label_mappings_dict


def ai_determine_leiden_resolution_adata_dict(adata_dict, initial_resolution=1):
    """
    Adjusts Leiden clustering resolution for each AnnData object in a dictionary based on AI feedback.

    Args:
        adata_dict (dict): Dictionary of AnnData objects.
        initial_resolution (float): Initial resolution for Leiden clustering (default is 1).

    Returns: dict: Dictionary with final resolution values after AI-based adjustments.
    """
    return adata_dict_fapply_return(adata_dict, ai_determine_leiden_resolution, max_retries=3, initial_resolution=initial_resolution)


def harmony_label_transfer_adata_dict(adata_dict, master_data, master_subset_column='tissue', label_column='cell_type'):
    """harmony label transfer each adata in an AdataDict"""
    adata_dict_fapply(adata_dict, harmony_label_transfer, master_data=master_data, master_subset_column=master_subset_column, label_column=label_column)


def plot_sankey_adata_dict(adata_dict, cols, params=None):
    """
    Applies plot_sankey to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, plot_sankey, cols=cols, params=params)


def save_sankey_adata_dict(plot_dict, filename):
    """
    Saves each sankey plot in a dictionary (i.e. the return value of plot_sankey_adata_dict)
    """
    adata_dict_fapply(plot_dict, save_sankey, filename=filename)

def plot_grouped_average_adata_dict(adata_dict, label_value):
    """
    plots the grouped average of a value for each group of a label. label_value must be a dictionary of dictionaries. For example, if adata_dict has two anndata with keys 'ad1' and 'ad2', then setting label_value = {'ad1':{'cell_type':'pct_counts_mt'}, 'ad2':{'cell_type':'pct_counts_mt'}} would plot the average of pct_counts_mt for each cell type in the anndata on separate plots for each anndata in adata_dict.
    """
    adata_dict_fapply(adata_dict, plot_grouped_average, label_value=label_value)


def plot_changes_adata_dict(adata_dict, true_label_key, predicted_label_key, percentage=True):
    """
    Applies the plot_final_mismatches function to each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    predicted_label_key (str): The key in obs for predicted labels.
    true_label_key (str): The key in obs for true labels.
    percentage (bool): If True, plot percentages, otherwise plot counts.
    """
    for stratum, adata in adata_dict.items():
        print(f"Plotting changes for {stratum}")
        plot_changes(adata, true_label_key, predicted_label_key, percentage, stratum)


def plot_confusion_matrix_adata_dict(adata_dict, true_label_key, predicted_label_key,
                                     row_color_keys=None, col_color_keys=None, figsize=(10,10), diagonalize=False):
    """
    Applies the plot_confusion_matrix_from_adata function to each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    true_label_key (str): The key in obs for true class labels.
    predicted_label_key (str): The key in obs for predicted class labels.
    title (str): Title of the plot, which will be prefixed with the stratum name.
    row_color_keys (list): Optional keys for row colors in adata.obs.
    col_color_keys (list): Optional keys for column colors in adata.obs.
    """
    for stratum, adata in adata_dict.items():
        # Customize title for each subset
        subset_title = f"Confusion Matrix for {stratum}"
        plot_confusion_matrix_from_adata(adata, true_label_key, predicted_label_key, title=subset_title,
                                         row_color_keys=row_color_keys, col_color_keys=col_color_keys, figsize=figsize, diagonalize=diagonalize)


def summarize_metadata_adata_dict(adata_dict, **kwargs):
    """
    Generate summary tables for each AnnData object in the dictionary using the summarize_metadata function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments, including 'columns' which specifies a list of columns from the metadata to summarize. Use '*' to specify joint frequencies of multiple columns.

    Returns:
    - dict: A dictionary of summary dictionaries for each AnnData object in the adata_dict.
    """
    return adata_dict_fapply_return(adata_dict, summarize_metadata, **kwargs)


def display_html_summary_adata_dict(summary_dict_dict):
    """
    Display separate HTML tables for each metadata category in the summary dictionaries,
    arranging up to three tables in a row before starting a new line.
    
    Parameters:
    - summary_dict_dict (dict): A dictionary of summary dictionaries for each AnnData object in the adata_dict.
    """
    for stratum, summary_dict in summary_dict_dict.items():
        print(f"Summary for {stratum}:")
        display_html_summary(summary_dict)


def pca_density_adata_dict(adata_dict, keys):
    """
    Applies PCA-based density filtering recursively on subsets of an AnnData dictionary. Each subset
    is determined by the provided keys. The function returns a dictionary where each AnnData object
    has an additional metadata key indicating the result of the density filter. The structure of the
    input dictionary is preserved, and each AnnData object's metadata is updated in-place.

    Parameters:
    - adata_dict: Dictionary of AnnData objects, with keys indicating different groups.
    - keys: List of keys to further stratify the AnnData objects if recursion is needed.

    Returns:
    - Dictionary: Updated adata_dict with the same keys but with each AnnData object having a new metadata key 'density_filter'.
    """
    

    if len(keys) == 0:
        # No further keys to split by, apply filtering directly
        for label, adata in adata_dict.items():
            X = adata.X
            if X.shape[0] < 10:
                density, cutoff = np.ones(X.shape[0]), 0
            else:
                density, cutoff, _ = pca_density_filter(
                    X, n_components=3, threshold=0.10
                )
            high_density_indices = density > cutoff
            index_vector = np.zeros(X.shape[0], dtype=bool)
            index_vector[high_density_indices] = True
            add_label_to_adata(
                adata, np.arange(X.shape[0]), index_vector, "density_filter"
            )
    else:
        # Recurse into further keys
        first_key = keys[0]
        new_keys = keys[1:]
        for label, adata in adata_dict.items():
            subgroups = build_adata_dict(
                adata, [first_key], {first_key: adata.obs[first_key].unique().tolist()}
            )
            pca_density_wrapper(subgroups, new_keys)  # Recursively update each subgroup
            # Combine results back into the original adata entry
            updated_adata = concatenate_adata_dict(subgroups)
            adata_dict[label] = updated_adata

    return adata_dict

# def pca_density_adata_dict(adata_dict, keys):
#     """
#     This function applies PCA-based density filtering to the AnnData objects within adata_dict.
#     If adata_dict contains only one key, the filtering is applied directly. If there are multiple keys,
#     it recursively builds new adata dictionaries for subsets based on the provided keys and applies
#     the filtering to these subsets. Finally, it concatenates the results back into a single AnnData object.

#     Parameters:
#     - adata_dict: Dictionary of AnnData objects, with keys indicating different groups.
#     - keys: List of keys to stratify the AnnData objects further if more than one group is present.

#     Returns:
#     - AnnData object containing the results of PCA density filtering applied to each subset,
#       with results combined if the initial dictionary had more than one key.
#     """
#     if len(adata_dict) == 1:
#         # Only one group in adata_dict, apply density filter directly
#         label, adata = next(iter(adata_dict.items()))
#         X = adata.X
#         if X.shape[0] < 10:
#             density, cutoff = np.ones(X.shape[0]), 0
#         else:
#             density, cutoff, _ = pca_density_filter(X, n_components=3, threshold=0.10)
#         high_density_indices = density > cutoff
#         index_vector = np.zeros(X.shape[0], dtype=bool)
#         index_vector[high_density_indices] = True
#         add_label_to_adata(adata, np.arange(X.shape[0]), index_vector, 'density_filter')
#         return adata
#     else:
#         # More than one group, handle recursively
#         first_key = keys[0]
#         new_keys = keys[1:]
#         updated_adatas = {}
#         for key, group_adata in adata_dict.items():
#             new_adata_dict = build_adata_dict(group_adata, new_keys, {k: group_adata.obs[k].unique().tolist() for k in new_keys})
#             updated_adatas[key] = pca_density_wrapper(new_adata_dict, new_keys)
#         return concatenate_adata_dict(updated_adatas)