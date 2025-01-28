"""
This subpackage contains plotting modules.
"""

from .cells import (
    module_score_barplot,
    module_score_umap,

)

from .genes import (
    annotate_gene_groups_with_ai_biological_process,
)

from .benchmarking import (
    plot_training_history,
    plot_changes,
    plot_confusion_matrix_from_adata,
    plot_confusion_matrix,
    plot_sankey,
    save_sankey,
    plot_grouped_average,
    plot_model_agreement,
    plot_model_agreement_categorical,
    
)

__all__ = [
    # from cells.py
    "module_score_barplot",
    "module_score_umap",

    # from genes.py
    "annotate_gene_groups_with_ai_biological_process",

    # from benchmarking.py
    "plot_training_history",
    "plot_changes",
    "plot_confusion_matrix_from_adata",
    "plot_confusion_matrix",
    "plot_sankey",
    "save_sankey",
    "plot_grouped_average",
    "plot_model_agreement",
    "plot_model_agreement_categorical",
]
