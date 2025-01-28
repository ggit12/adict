"""
This module contains functions that use LLMs to automate label management within and between dataframes.
All of these functions are for processing category labels in pd dfs.
"""

#post-process single categorical columns
from .clean_single_column import (
    simplify_obs_column,
    simplify_var_index,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,
    map_gene_labels_to_simplified_set,

)

#make categorical columns within a df share a common set of labels
from .unify_columns_within_adata import (
    ensure_label_consistency_main,
    ensure_label_consistency_adata,

)

#make categorical columns between dfs share a common set of labels
from .unify_columns_between_adata import (
    ai_unify_labels
)

__all__ = [
    # from clean_single_column.py
    "simplify_obs_column",
    "simplify_var_index",
    "create_label_hierarchy",
    "map_cell_type_labels_to_simplified_set",
    "map_gene_labels_to_simplified_set",

    # from unify_columns_within_adata.py
    "ensure_label_consistency_main",
    "ensure_label_consistency_adata",

    # from unify_columns_between_adata.py
    "ai_unify_labels",
]
