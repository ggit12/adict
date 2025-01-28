"""
This subpackage contains functions to compare labels.
"""

#label comparison utility functions
from .label_utils import (
   create_label_df,
)

#functions to calculate agreement between label columns
from .agreement_metrics import (
    kappa_adata,
    krippendorff_alpha_adata,
)

#ai methods to compare labels
from .ai_compare_cell_type_labels import (
   ai_compare_cell_type_labels_pairwise,
)


__all__ = [
    # label_utils.py
    "create_label_df",

    # agreement_metrics.py
    "kappa_adata",
    "krippendorff_alpha_adata",

    # ai_compare_cell_type_labels.py
    "ai_compare_cell_type_labels_pairwise",
]