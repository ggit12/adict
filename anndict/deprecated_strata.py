#functions to process an anndata across strata given by key(s), might be redundant with respect to adata_dict functions
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import re
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .dict import check_and_create_strata

from .stablelabel import (
    get_slurm_cores,
    pca_density_filter,
    pca_density_wrapper,
    pca_density_adata_dict,
    stable_label,
    stable_label_adata,
    update_adata_labels_with_results,
    plot_training_history,
    plot_changes,
    plot_confusion_matrix_from_adata,
    plot_confusion_matrix
)

def preprocess_adata_strata(adata, strata_keys, target_cells = 10000, min_cells = 25):
    """
    Preprocess an AnnData object by stratifying, filtering, and subsampling based on specified criteria.

    Parameters:
    adata (AnnData): Annotated data matrix.
    strata_keys (list of str): List of column names in `adata.obs` to use for stratification.
    target_cells (int, optional): Target number of cells to retain per stratum. Default is 10000.
    min_cells (int, optional): Minimum number of cells required to retain a stratum. Default is 25.

    Returns:
    AnnData: Concatenated AnnData object after filtering and subsampling.

    Raises:
    ValueError: If no strata meet the minimum cell requirement after filtering.
    """
    # Check and create stratfying variable in adata
    strata_key = check_and_create_strata(adata, strata_keys)

    # Filter cell types and subsample if necessary
    adatas = []
    for stratum in adata.obs[strata_key].cat.categories:
        subset = adata[adata.obs[strata_key] == stratum]
        if subset.n_obs > target_cells:
            subset = sc.pp.subsample(subset, n_obs=target_cells, copy=True)
        if subset.n_obs >= min_cells:
            adatas.append(subset)
    
    # Check if there's at least one valid cell type left after filtering
    if not adatas:
        raise ValueError("No cell types with the minimum required cells found.")

    # Concatenate the list of AnnData objects into a single AnnData object using anndata.concat
    adata_downsampled = ad.concat(adatas, join='outer')

    return adata_downsampled


def stable_label_adata_strata(adata, feature_key, label_key, strata_keys, classifier, max_iterations=100, stability_threshold=0.05, moving_average_length=3, random_state=None):
    """
    Trains a classifier for each stratum specified in adata.obs using a strata_key.

    Parameters:
    - adata: AnnData object containing the dataset.
    - feature_key: str, key to access the features in adata.obsm.
    - label_key: str, key to access the labels in adata.obs.
    - strata_key: str, key to differentiate strata in adata.obs.
    - classifier: classifier instance that implements fit and predict_proba methods.
    - max_iterations, stability_threshold, moving_average_length, random_state: passed directly to train_classifier_with_categorical_labels.

    Returns:
    - results: dict, keys are strata labels and values are lists containing the outputs from train_classifier_with_categorical_labels.
    """
    
    # Check and create stratfying variable in adata
    strata_key = check_and_create_strata(adata, strata_keys)

    # Determine unique strata
    strata = np.unique(adata.obs[strata_key])

    # Initialize results dictionary
    stable_label_results = {}

    for stratum in strata:
        # Subset adata for the current stratum
        subset_adata = adata[adata.obs[strata_key] == stratum].copy()

        # Capture indices of the subset
        indices = np.array(subset_adata.obs.index)
        
        # Train classifier on this subset
        trained_classifier, history, iterations, final_labels, label_encoder = stable_label_adata(
            subset_adata, feature_key, label_key, classifier, max_iterations, stability_threshold, moving_average_length, random_state
        )
        
        # Store results in dictionary
        stable_label_results[stratum] = {
            'classifier': trained_classifier,
            'history': history,
            'iterations': iterations,
            'final_labels': final_labels,
            'label_encoder': label_encoder,
            'indices': indices
        }

    return stable_label_results

def plot_confusion_matrix_across_strata(adata, true_label_key, predicted_label_key, strata_keys, title='Confusion Matrix'):
    """
    Wrapper function to plot confusion matrices for different strata in an AnnData object.

    Parameters:
    - adata: AnnData object containing the dataset.
    - true_label_key: str, key to access the true class labels in adata.obs.
    - predicted_label_key: str, key to access the predicted class labels in adata.obs.
    - strata_keys: list of str, keys to use for stratifying the data.
    - title: str, prefix to the title of the plot for each stratum.
    """
    # Check and create stratifying variable in adata
    strata_key = check_and_create_strata(adata, strata_keys)

    # Determine unique strata
    strata = np.unique(adata.obs[strata_key])

    for stratum in strata:
        # Subset adata for the current stratum
        subset_adata = adata[adata.obs[strata_key] == stratum].copy()

        # Update title with stratum
        stratum_title = f"{title} for {stratum}"

        # Plot confusion matrix for the subset
        plot_confusion_matrix_from_adata(subset_adata, true_label_key, predicted_label_key, stratum_title, row_color_keys=None, col_color_keys=None)