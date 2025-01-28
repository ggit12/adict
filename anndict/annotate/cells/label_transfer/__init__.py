"""
This subpackage contains functions to transfer labels (from an already labeled adata to an unlabeled adata).
"""

#label transfer based on harmony integration
from .harmony import (
    harmony_label_transfer,
)

#label transfer by training a classifier on labeled data, then running the classifier on unlabeled data
from .stabilizing_classifier import (
    #still need to rework stablelabel funcs
    stable_label,
    stable_label_adata,
)


__all__ = [
    # harmony.py
    "harmony_label_transfer",

    # stabilizing_classifier.py
    "stable_label",
    "stable_label_adata",
]