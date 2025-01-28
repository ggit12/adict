"""
The module contains functions to annotate cell subtype by comparison.
Specifically, this means asking an LLM to annotate a subtype by comparing its marker genes
to the marker genes of other subtypes of that cell type).
"""

from .base import ai_annotate_by_comparison