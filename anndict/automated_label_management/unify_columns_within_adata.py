"""
This module contains functions that operate on a single adata to make multiple columns in the .obs have a shared set of categories.
"""

#the following functions also unify labels but serve a different purpose than ai_unify_labels.
#ai_unify_labels is meant to unify labels across multiple adata
#the following set of ensure_label functions are meant to operate within a single adata
#and do not communicate across multiple adata in a dict

from anndict.utils import normalize_label
from anndict.automated_label_management.clean_single_column import map_cell_type_labels_to_simplified_set

def ensure_label_consistency_adata(adata, cols, simplification_level='unified, typo-fixed', new_col_prefix='consistent'):
    """
    Wrapper function to ensure label consistency across specified columns in an AnnData object.
    
    Parameters:
    - adata: AnnData object
    - cols: List of column names in adata.obs to ensure label consistency
    - simplification_level: Level of simplification for label mapping
    - new_col_prefix: Prefix to create new columns in adata.obs. Default is "" (overwrites original columns).

    Returns:
    - Updated adata with consistent labels in adata.obs[new_col_prefix + cols]
    - label_map: Dictionary mapping original labels to the simplified labels
    """
    # Step 1: Extract the relevant columns from adata.obs into a DataFrame
    df = adata.obs[cols].copy()

    # Step 2: Ensure label consistency using the helper function
    consistent_df, label_map = ensure_label_consistency_main(df, simplification_level)

    # Step 3: Create new columns in adata.obs with the prefix
    for col in cols:
        new_col_name = f"{new_col_prefix}_{col}"
        adata.obs[new_col_name] = consistent_df[col]

    return label_map


def ensure_label_consistency_main(df, simplification_level='unified, typo-fixed'):
    """
    Function to ensure label consistency across multiple columns in a DataFrame
    by mapping labels to a unified and simplified set.
    """
    # Step 1: Normalize all labels in the DataFrame
    for column in df.columns:
        df[column] = df[column].apply(normalize_label)

    # Step 2: Create a unified set of unique labels across all columns
    unique_labels = set()
    for column in df.columns:
        unique_labels.update(df[column].unique())

    # Step 3: Use the external function to map labels to a simplified set
    unique_labels_list = list(unique_labels)
    mapping_dict = map_cell_type_labels_to_simplified_set(unique_labels_list, simplification_level=simplification_level)

    # Step 4: Apply the mapping dictionary to all columns
    for column in df.columns:
        df[column] = df[column].map(mapping_dict)

    return df, mapping_dict