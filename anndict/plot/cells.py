"""
This module contains visualizations functions for cell typing.
"""
import numpy as np
import scanpy as sc


import matplotlib.pyplot as plt


def module_score_barplot(adata, group_cols, score_cols, adt_key=None, figsize=(10,8)):
    """
    Create a bar plot of mean module scores grouped by specified columns.

    Parameters:
    adata : AnnData
        The AnnData object containing the data.
    group_cols : str or list of str
        The column(s) in adata.obs to group by.
    score_cols : str or list of str
        The column(s) in adata.obs that contain the module scores.

    Returns:
    fig, ax : matplotlib Figure and Axes
        The figure and axes objects of the plot.
    """
    #print adt_key if provided
    if adt_key:
        print(adt_key)

    # Ensure group_cols and score_cols are lists
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    # Select module score columns and the group columns
    module_scores = adata.obs[score_cols + group_cols]

    # Group by the group_cols and compute mean module scores
    mean_scores = module_scores.groupby(group_cols, observed=False).mean()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the mean module scores as a grouped bar plot
    mean_scores.plot(kind='bar', ax=ax)

    # Set labels, title, and legend location
    ax.set_ylabel('Mean Module Score')
    ax.legend(title=None, loc=6, bbox_to_anchor=(1,0.5))

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax


def module_score_umap(adata, score_cols, adt_key=None, **kwargs):
    """
    Generates UMAP plots for specified module scores in a single figure.

    Parameters:
        adata (AnnData): Annotated data matrix containing UMAP coordinates and module scores.
        score_cols (list of str): List of column names in `adata` containing module scores to plot.
        **kwargs: Additional keyword arguments passed to `sc.pl.umap`, including 'vmax' for color scaling 
                  (default is 'p99').

    Returns:
        matplotlib.figure.Figure: Figure containing the UMAP plots.
    """
    # Print adt_key if provided
    if adt_key:
        print(adt_key)

    # Extract vmax from kwargs or default to 'p99'
    vmax = kwargs.pop('vmax', 'p99')

    n_plots = len(score_cols)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Ensure axes is a flat array of axes
    axes = np.atleast_1d(axes).ravel()

    for i, (score_col, ax) in enumerate(zip(score_cols, axes)):
        # Process title
        title = ' '.join(word.capitalize() for word in score_col.replace('_', ' ').split())
        # Plot the UMAP
        sc.pl.umap(adata, color=score_col, title=title, vmax=vmax, ax=ax, show=False, **kwargs)
        ax.set_title(title)

    # Turn off any unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    return fig
