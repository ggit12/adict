"""
This subpackage contains functions to annotate and assess cell and gene annotation.
"""

#gene annotation functions
from .genes import (
    ai_gene_list
)

#cell annotation functions
from .cells import (
    # De Novo
    ai_annotate, ai_annotate_by_comparison,
    ai_cell_type, ai_annotate_cell_type,
    ai_cell_types_by_comparison,
    ai_annotate_cell_sub_type,
    ai_annotate_biological_process,
    cell_type_marker_gene_score,
    ai_determine_leiden_resolution,

    # Label Transfer
    harmony_label_transfer,
    stable_label, stable_label_adata,

    # Benchmarking
    create_label_df,
    kappa_adata, krippendorff_alpha_adata,
    ai_compare_cell_type_labels_pairwise
)

__all__ = [
    # Genes
    #--------
    # genes.py
    "ai_gene_list",

    # Cells
    #--------
    # De Novo
    "ai_annotate",
    "ai_annotate_by_comparison",
    "ai_cell_type",
    "ai_annotate_cell_type",
    "ai_cell_types_by_comparison",
    "ai_annotate_cell_sub_type",
    "ai_annotate_biological_process",
    "cell_type_marker_gene_score",
    "ai_determine_leiden_resolution",

    # Label Transfer
    "harmony_label_transfer",
    "stable_label",
    "stable_label_adata",

    # Benchmarking
    "create_label_df",
    "kappa_adata",
    "krippendorff_alpha_adata",
    "ai_compare_cell_type_labels_pairwise"
]