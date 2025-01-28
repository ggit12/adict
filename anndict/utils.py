# functions that aren't part of the stablelabel pipeline and operate on a single anndata (unstratified)
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from IPython.display import HTML, display


def enforce_semantic_list(lst):
    """This function runs a number of checks to make sure that the input is a semantic list, and not i.e. integers cast as strings."""
    error_message = "input list appears to contain any of: NaN, numeric, or numeric cast as string. Please ensure you are passing semantic labels (i.e. gene symbols or cell types) and not integer labels for AI interpretation. Make sure adata.var.index and adata.obs.index are not integers or integers cast as strings."

    def get_context(lst, index):
        before = lst[index - 1] if index > 0 else None
        after = lst[index + 1] if index < len(lst) - 1 else None
        return before, after

    # Check if all items are strings
    for index, item in enumerate(lst):
        if not isinstance(item, str):
            before, after = get_context(lst, index)
            raise ValueError(
                f"{error_message} Item at index {index} is not a string: {item}. Context: Before: {before}, After: {after}"
            )

    # Check if any item can be converted to float
    for index, item in enumerate(lst):
        try:
            float(item)
        except ValueError:
            pass
        else:
            before, after = get_context(lst, index)
            raise ValueError(
                f"{error_message} Item at index {index} can be cast to a number: {item}. Context: Before: {before}, After: {after}"
            )

    return True


def make_names(names):
    """
    Convert a list of names into valid and unique Python identifiers.

    Args:
        names (list of str): List of names to be transformed.

    Returns:
        list of str: Valid and unique Python identifiers.
    """
    # Equivalent of R's make.names() function in Python
    valid_names = []
    seen = {}
    for name in names:
        # Replace invalid characters with underscores
        clean_name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}.{seen[clean_name]}"
        else:
            seen[clean_name] = 0
        valid_names.append(clean_name)
    return valid_names


def normalize_string(s):
    """Removes non-alphanumeric characters and converts to lowercase."""
    return re.sub(r"[^\w\s]", "", s.lower())


def normalize_label(label):
    """
    Calls normalize-string and handles NaN values.
    """
    if pd.isna(label):  # Handle NaN values
        return "missing"
    return normalize_string(label.strip())


def create_color_map(adata, keys):
    """
    Creates a unified color map for given keys from an AnnData object, differentiating
    between continuous and categorical data.

    Parameters:
    - adata: AnnData object.
    - keys: list of str, keys for which the color map is required.

    Returns:
    - dict: A color map linking unique values or ranges from the specified keys to colors.
    """
    color_map = {}
    for key in keys:
        if pd.api.types.is_numeric_dtype(adata.obs[key]):
            # Create a continuous colormap
            min_val, max_val = adata.obs[key].min(), adata.obs[key].max()
            norm = plt.Normalize(min_val, max_val)
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
            # Store the scalar map directly
            color_map[key] = scalar_map
        else:
            # Handle as categorical
            unique_values = pd.unique(adata.obs[key])
            color_palette = sns.color_palette("husl", n_colors=len(unique_values))
            color_palette_hex = [
                matplotlib.colors.rgb2hex(color) for color in color_palette
            ]
            color_map[key] = dict(zip(unique_values, color_palette_hex))

    return color_map


def get_slurm_cores():
    """
    Returns the total number of CPU cores allocated to a Slurm job based on environment variables.
    """
    # Get the number of CPUs per task (default to 1 if not set)
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))

    # Get the number of tasks (default to 1 if not set)
    ntasks = int(os.getenv("SLURM_NTASKS", "1"))

    # Calculate total cores
    total_cores = cpus_per_task * ntasks

    return total_cores


def pca_density_filter(data, n_components=3, threshold=0.10):
    """
    Calculate density contours for PCA-reduced data, return the density of all input data,
    and identify the unique variables that were included in the PCA.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
    - n_components: int, number of components for PCA to reduce the data to.

    Returns:
    - pca_data: PCA-reduced data (None if all variables are constant).
    - density: Density values of all the points (None if all variables are constant).
    - unique_variables: List of unique variables that were included in the PCA (empty list if all variables are constant).
    """

    # Check for constant variables (these will not be used by PCA)
    non_constant_columns = np.var(data, axis=0) > 0

    # Skip the block if no non-constant variables are found
    if not np.any(non_constant_columns):
        return None, None, []

    # Adjust n_components if necessary
    n_features = np.sum(non_constant_columns)
    n_samples = data.shape[0]
    n_components = min(n_components, n_features, n_samples)

    unique_variables = np.arange(data.shape[1])[non_constant_columns]

    # Perform PCA reduction only on non-constant variables
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data[:, non_constant_columns])

    # Calculate the point density for all points
    kde = gaussian_kde(pca_data.T)
    density = kde(pca_data.T)

    # Determine the density threshold
    cutoff = np.percentile(density, threshold * 100)

    return density, cutoff, unique_variables.tolist()


def pca_density_wrapper(X, labels):
    """
    Apply calculate_density_contours_with_unique_variables to subsets of X indicated by labels.
    Returns a vector indicating whether each row in X is above the threshold for its respective label group.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels: array-like, shape (n_samples,), labels indicating the subset to which each row belongs

    Returns:
    - index_vector: array-like, boolean vector of length n_samples indicating rows above the threshold
    """
    unique_labels = np.unique(labels)
    index_vector = np.zeros(len(X), dtype=bool)

    for label in unique_labels:
        subset = X[labels == label]
        if subset.shape[0] < 10:
            # If fewer than 10 cells, include all cells by assigning density = 1 and cutoff = 0
            density, cutoff = np.ones(subset.shape[0]), 0
        else:
            density, cutoff, _ = pca_density_filter(
                subset, n_components=3, threshold=0.10
            )

        # Mark rows above the threshold for this label
        high_density_indices = density > cutoff
        global_indices = np.where(labels == label)[0][high_density_indices]
        index_vector[global_indices] = True

    return index_vector


def display_html_summary(summary_dict):
    """
    Display separate HTML tables for each metadata category in the summary dictionary,
    arranging up to three tables in a row before starting a new line.

    Parameters:
        summary_dict (dict): The dictionary containing frequency data for metadata columns.
    """
    html = '<div style="display: flex; flex-wrap: wrap;">'
    table_count = 0

    for category, data in summary_dict.items():
        if table_count % 3 == 0 and table_count != 0:
            html += '<div style="flex-basis: 100%; height: 20px;"></div>'

        table_html = f'<div style="flex: 1; padding: 10px;"><h3>{category}</h3>'
        # Start the table and add a header row
        table_html += '<table border="1"><tr><th></th>'  # Empty header for the row labels
        table_html += ''.join(f'<th>{col}</th>' for col in data.columns) + '</tr>'  # Column headers

        for index, row in data.iterrows():
            # Include row labels as the first column and the rest of the data in subsequent columns
            table_html += f'<tr><td>{index}</td>' + ''.join(f'<td>{val}</td>' for val in row) + '</tr>'

        table_html += '</table></div>'
        html += table_html
        table_count += 1

    html += '</div>'
    display(HTML(html))
