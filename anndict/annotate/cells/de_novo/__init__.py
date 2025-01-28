"""
This subpackage contains functions to annotate cells de novo (from scratch), based on marker genes.
"""

#the base ai_annotate functions
from .base import (
    ai_annotate,
    ai_annotate_by_comparison
)

#cell annotation methods--by gene list
from .annotate_cell_type import (
    ai_cell_type,
    ai_annotate_cell_type
)

#cell type annotation methods--by gene list (in the context of other gene lists)
from .annotate_cell_type_by_comparison import (
    ai_cell_types_by_comparison,
)

from .annotate_cell_subtype import (
    ai_annotate_cell_sub_type,
)

#annotate groups of cells with biological process (and not cell type)
from .annotate_biological_process import (
    ai_annotate_biological_process,
)

#automatically calculate gene module scores for each cell type
from .cell_type_score import (
    cell_type_marker_gene_score,
)

#automatically determine cluster resolution
from .automated_clustering import (
    ai_determine_leiden_resolution,
)

__all__ = [
    # base.py
    "ai_annotate",
    "ai_annotate_by_comparison",

    # annotate_cell_type.py
    "ai_cell_type",
    "ai_annotate_cell_type",

    # annotate_cell_type_by_comparison.py
    "ai_cell_types_by_comparison",

    # annotate_cell_subtype.py
    "ai_annotate_cell_sub_type",

    # annotate_biological_process.py
    "ai_annotate_biological_process",

    # cell_type_score.py
    "cell_type_marker_gene_score",

    # automated_clustering.py
    "ai_determine_leiden_resolution"
]
