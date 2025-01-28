"""
This module contains functions to calculate reliability metrics on columns adata.
"""
import numpy as np
import krippendorff

from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.inter_rater import fleiss_kappa

from anndata import AnnData


def kappa_adata(adata: AnnData,
cols: list[str]
) -> dict:
    """
    Calculate pairwise Cohen's Kappa, average pairwise Kappa, and Fleiss' Kappa
    for categorical data in ``adata.obs[cols]``.

    Parameters
    ------------
    adata
        An :class:`AnnData` object.

    cols
        List of columns in ``adata.obs`` to use for calculating agreement.

    Returns
    -------
    dict
        A :class:`dictionary` with the following keys:

        - ``pairwise``: A :class:`dictionary` with pairwise Cohen's Kappa values.
        - ``average_pairwise``: A dictionary with the average pairwise Kappa for each rater.
        - ``fleiss``: The Fleiss' Kappa value for the overall agreement across all raters.
    """

    # Extract data from adata.obs based on the specified columns
    data = adata.obs[cols].to_numpy()
    num_raters = len(cols)
    kappa_scores = {'pairwise': {}, 'average_pairwise': {}, 'fleiss': None}

    # Calculate pairwise Cohen's Kappa
    for i in range(num_raters):
        rater_kappas = []
        for j in range(num_raters):
            if i != j:
                # Calculate Cohen's Kappa for each pair
                kappa = cohen_kappa_score(data[:, i], data[:, j])
                kappa_scores['pairwise'][(cols[i], cols[j])] = kappa
                rater_kappas.append(kappa)

        # Average Kappa for this rater (with every other rater)
        avg_kappa = np.mean(rater_kappas) if rater_kappas else None
        kappa_scores['average_pairwise'][cols[i]] = avg_kappa

    # Fleiss' Kappa calculation
    unique_categories = np.unique(data)
    category_map = {cat: idx for idx, cat in enumerate(unique_categories)}
    fleiss_data = np.zeros((data.shape[0], len(unique_categories)))

    # Count category occurrences per item (per row) using vectorized operations
    for i in range(data.shape[0]):
        row = np.array([category_map[val] for val in data[i]])
        fleiss_data[i] = np.bincount(row, minlength=len(unique_categories))

    # Calculate Fleiss' Kappa
    fleiss_kappa_value = fleiss_kappa(fleiss_data)
    kappa_scores['fleiss'] = fleiss_kappa_value

    return kappa_scores


def krippendorff_alpha_adata(
    adata: AnnData,
    cols: list[str],
    level_of_measurement: str = 'nominal'
) -> float:
    """
    Calculate Krippendorff's Alpha for categorical data in ``adata.obs[cols]``.

    Parameters
    ------------
    adata
        An :class:`AnnData` object.

    cols
        List of columns in ``adata.obs`` to use for calculating agreement.

    level_of_measurement
        The type of data (``'nominal'``, ``'ordinal'``, ``'interval'``, ``'ratio'``).
        Default is ``'nominal'`` (for categorical data).

    Returns
    -------
    float
        Krippendorff's Alpha for the specified columns in ``adata.obs``.
    """
    # Extract data from adata.obs based on the specified columns
    data = adata.obs[cols].to_numpy()

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Flatten the data, fit the encoder, and reshape back
    flat_data = data.ravel()
    encoded_flat = le.fit_transform(flat_data)
    encoded_data = encoded_flat.reshape(data.shape)

    # Transpose the data to match Krippendorff's alpha input format
    # (units as columns, raters as rows)
    encoded_data = encoded_data.T

    # Calculate Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data=encoded_data, level_of_measurement=level_of_measurement)

    return alpha
