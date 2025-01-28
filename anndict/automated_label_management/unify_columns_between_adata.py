"""
This module contains functions that operate across multiple adata, to make a sure that each adata has a .obs column with a set of categories that is shared between all the adata.
"""

from anndict.automated_label_management.clean_single_column import map_cell_type_labels_to_simplified_set

def ai_unify_labels(adata_dict, label_columns, new_label_column, simplification_level='unified, typo-fixed'):
    """
    Unifies cell type labels across multiple AnnData objects by mapping them to a simplified, unified set of labels.

    Parameters:
    adata_dict (dict): Dictionary where keys are identifiers and values are AnnData objects.
    label_columns (dict): Dictionary where keys should be the same as the keys of adata_dict and values are the column names in .obs containing the original labels.
    new_label_column (str): Name of the new column to be created in .obs for storing the harmonized labels.

    Returns:
    dict: A mapping dictionary where the keys are the original labels and the values are the unified labels.
    """
    #todo: use adata_dict_fapply instead of loops
    def get_unique_labels_from_obs_column(adata, label_column):
        return adata.obs[label_column].unique().tolist()

    def apply_mapping_to_adata(adata, mapping_dict, original_column, new_column, adt_key=None):
        adata.obs[new_column] = adata.obs[original_column].map(mapping_dict)

    # Step 1: Aggregate all labels
    # aggregated_labels = adata_dict_fapply_return(adata_dict, get_unique_labels_from_obs_column, )
    aggregated_labels = []
    for key in adata_dict:
        labels = get_unique_labels_from_obs_column(adata_dict[key], label_columns[key])
        aggregated_labels.extend(labels)
    unique_labels_list = list(set(aggregated_labels))

    # Step 2: Get the mapping dictionary
    mapping_dict = map_cell_type_labels_to_simplified_set(unique_labels_list, simplification_level=simplification_level)

    # Step 3: Apply the mapping to each anndata in adata_dict
    for key in adata_dict:
        apply_mapping_to_adata(adata_dict[key], mapping_dict, label_columns[key], new_label_column)

    return mapping_dict