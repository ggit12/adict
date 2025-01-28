"""
This module makes plots of the results of benchmarking (i.e. if you already have a ground truth label column in your dataset, and want to compare predictions).
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from anndict.utils import create_color_map

def plot_training_history(results, separate=True):
    """
    Plot the training history of a model, showing percent label change versus iteration.

    Parameters:
    results (dict): Dictionary where keys are strata names and values are dictionaries containing training history.
    separate (bool, optional): If True, plot each stratum's training history separately. If False, plot all strata together. Default is True.

    Returns:
    None
    """
    if separate:
        for stratum, info in results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(info['history'], marker='o')
            plt.title(f'Percent Label Change vs. Iteration - {stratum}')
            plt.xlabel('Iteration')
            plt.ylabel('Percent Label Change')
            plt.grid(True)
            plt.show()
    else:
        plt.figure(figsize=(10, 6))
        for stratum, info in results.items():
            plt.plot(info['history'], marker='.', label=stratum)
        plt.title('Percent Label Change vs. Iteration - All Strata')
        plt.xlabel('Iteration')
        plt.ylabel('Percent Label Change')
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_changes(adata, true_label_key, predicted_label_key, percentage=True, stratum=None):
    """
    Plot the changes between true and predicted labels in an AnnData object.

    Parameters:
    adata (AnnData): Annotated data matrix.
    true_label_key (str): Key for the true labels in adata.obs.
    predicted_label_key (str): Key for the predicted labels in adata.obs.
    percentage (bool, optional): If True, plot the percentage of labels changed. If False, plot the count of labels changed. Default is True.
    stratum (str, optional): Title for the plot, often used to indicate the stratum. Default is None.

    Returns:
    None
    """
    # Extract the series from the AnnData object's DataFrame
    data = adata.obs[[predicted_label_key, true_label_key]].copy()
    
    # Convert to categorical with a common category set
    common_categories = list(set(data[true_label_key].cat.categories).union(set(data[predicted_label_key].cat.categories)))
    data[true_label_key] = data[true_label_key].cat.set_categories(common_categories)
    data[predicted_label_key] = data[predicted_label_key].cat.set_categories(common_categories)
    
    # Add a mismatch column that checks whether the predicted and true labels are different
    data['Changed'] = data[true_label_key] != data[predicted_label_key]
    
    # Group by true label key and calculate the sum of mismatches or the mean if percentage
    if percentage:
        change_summary = data.groupby(true_label_key)['Changed'].mean()
    else:
        change_summary = data.groupby(true_label_key)['Changed'].sum()
    
    # Sort the summary in descending order
    change_summary = change_summary.sort_values(ascending=False)
    
    # Plotting
    ax = change_summary.plot(kind='bar', color='red', figsize=(10, 6))
    ax.set_xlabel(true_label_key)
    ax.set_ylabel('Percentage of Labels Changed' if percentage else 'Count of Labels Changed')
    ax.set_title(stratum)
    ax.set_xticklabels(change_summary.index, rotation=90)
    plt.xticks(fontsize=8)
    plt.show()


def plot_confusion_matrix_from_adata(adata, true_label_key, predicted_label_key, title='Confusion Matrix',
                                     row_color_keys=None, col_color_keys=None, figsize=None, diagonalize=False,
                                     true_ticklabels=None, predicted_ticklabels=None, annot=None):
    """
    Wrapper function to plot a confusion matrix from an AnnData object, with optional row and column colors.
    
    Parameters:
    - adata: AnnData object containing the dataset.
    - true_label_key: str, key to access the true class labels in adata.obs.
    - predicted_label_key: str, key to access the predicted class labels in adata.obs.
    - title: str, title of the plot.
    - row_color_key: str, key for row colors in adata.obs.
    - col_color_key: str, key for column colors in adata.obs.
    """

    # Check and convert row_color_key and col_color_key to lists if they are not None
    if row_color_keys is not None and not isinstance(row_color_keys, list):
        row_color_keys = [row_color_keys]

    if col_color_keys is not None and not isinstance(col_color_keys, list):
        col_color_keys = [col_color_keys]

    # Get unique labels
    true_labels = adata.obs[true_label_key].astype(str)
    predicted_labels = adata.obs[predicted_label_key].astype(str)

    combined_labels = pd.concat([true_labels, predicted_labels])
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_labels)

    #Encode labels
    true_labels_encoded = label_encoder.transform(true_labels)
    predicted_labels_encoded = label_encoder.transform(predicted_labels)

    # Create label-to-color dictionary for mapping
    true_label_color_dict = None
    if row_color_keys:
        true_label_subset = adata.obs[[true_label_key] + row_color_keys].drop_duplicates().set_index(true_label_key)
        true_label_color_dict = {label: {key: row[key] for key in row_color_keys}
                        for label, row in true_label_subset.iterrows()
                        }

    predicted_label_color_dict = None
    if col_color_keys:
        predicted_label_subset = adata.obs[[predicted_label_key] + col_color_keys].drop_duplicates().set_index(predicted_label_key)
        predicted_label_color_dict = {label: {key: col[key] for key in col_color_keys}
                        for label, col in predicted_label_subset.iterrows()
                        }

    # Compute the row and column colors
    # Get unified color mapping
    keys = list(set(row_color_keys or []).union(col_color_keys or []))
    color_map = create_color_map(adata, keys)

    # Call the main plot function
    return plot_confusion_matrix(true_labels_encoded, predicted_labels_encoded, label_encoder, color_map, title,
                          row_color_keys=row_color_keys, col_color_keys=col_color_keys,
                          true_label_color_dict=true_label_color_dict, predicted_label_color_dict=predicted_label_color_dict,
                          true_labels=true_labels, predicted_labels=predicted_labels, figsize=figsize, diagonalize=diagonalize,
                          true_ticklabels=true_ticklabels, predicted_ticklabels=predicted_ticklabels, annot=annot)


def plot_confusion_matrix(true_labels_encoded, predicted_labels_encoded, label_encoder, color_map, title='Confusion Matrix', 
                          row_color_keys=None, col_color_keys=None,
                          true_label_color_dict=None, predicted_label_color_dict=None,
                          true_labels=None, predicted_labels=None, figsize=None,
                          diagonalize=False, true_ticklabels=None, predicted_ticklabels=None, annot=None):
    """
    Plots a normalized confusion matrix with optional auto-diagonalization, clustering, and color annotations.

    Parameters:
        true_labels_encoded (array-like): Encoded true labels of the data.
        predicted_labels_encoded (array-like): Encoded predicted labels of the data.
        label_encoder (LabelEncoder): A fitted label encoder used to decode the labels.
        color_map (dict): A nested dictionary specifying colors for keys used in annotations.
        title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
        row_color_keys (list of str, optional): Keys from `color_map` to apply to rows based on `true_labels`.
        col_color_keys (list of str, optional): Keys from `color_map` to apply to columns based on `predicted_labels`.
        true_label_color_dict (dict, optional): Mapping of true labels to color information.
        predicted_label_color_dict (dict, optional): Mapping of predicted labels to color information.
        true_labels (array-like, optional): Original true labels (used for mapping row colors).
        predicted_labels (array-like, optional): Original predicted labels (used for mapping column colors).
        figsize (tuple, optional): Dimensions of the figure (width, height) in inches.
        diagonalize (bool, optional): If True, reorders the confusion matrix to make it as diagonal as possible. Default is False.
        true_ticklabels (list, optional): Custom tick labels for the true labels axis. Defaults to automatic handling.
        predicted_ticklabels (list, optional): Custom tick labels for the predicted labels axis. Defaults to automatic handling.
        annot (bool, optional): Whether to annotate the confusion matrix cells. Defaults to True for small matrices.

    Returns:
        sns.ClusterGrid: A seaborn ClusterGrid object representing the confusion matrix.

    Notes:
        - If the number of true or predicted labels exceeds 40, tick labels and annotations are disabled by default for better visibility.
        - Color annotations (row and column colors) are only applied if `row_color_keys` or `col_color_keys` are provided.
        - If `diagonalize` is True, the matrix is reordered using linear sum assignment to maximize diagonal dominance.
    """

    labels_true = np.unique(true_labels_encoded)
    labels_pred = np.unique(predicted_labels_encoded)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels_encoded, predicted_labels_encoded, labels=np.arange(len(label_encoder.classes_)))

    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = pd.DataFrame(cm_normalized[np.ix_(labels_true, labels_pred)], 
                                 index=label_encoder.inverse_transform(labels_true), 
                                 columns=label_encoder.inverse_transform(labels_pred))

    if diagonalize:
        # Sorting the confusion matrix to make it as diagonal as possible
        cost_matrix = -cm_normalized.values  # We need to minimize the cost, hence the negative sign
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Concatenate the optimal indices with the non-optimal ones
        row_ind = np.concatenate((row_ind, np.setdiff1d(np.arange(cm_normalized.shape[0]), row_ind)))
        col_ind = np.concatenate((col_ind, np.setdiff1d(np.arange(cm_normalized.shape[1]), col_ind)))

        cm_normalized = cm_normalized.iloc[row_ind, col_ind]
        labels_true_sorted = label_encoder.inverse_transform(labels_true)[row_ind]
        labels_pred_sorted = label_encoder.inverse_transform(labels_pred)[col_ind]
    else:
        labels_true_sorted = label_encoder.inverse_transform(labels_true)
        labels_pred_sorted = label_encoder.inverse_transform(labels_pred)

    def map_labels_to_colors(labels, label_color_dict, color_map):
        color_list = []
        for label in labels:
            color_dict = label_color_dict.get(label, {})
            colors = [color_map.get(key).get(color_dict.get(key, None), '#FFFFFF') for key in color_map]
            color_list.append(colors)
        return color_list

    row_colors = None
    if row_color_keys:
        row_colors = map_labels_to_colors(labels_true_sorted, true_label_color_dict, color_map)
        row_colors = pd.DataFrame(row_colors, index=labels_true_sorted)

    col_colors = None
    if col_color_keys:
        col_colors = map_labels_to_colors(labels_pred_sorted, predicted_label_color_dict, color_map)
        col_colors = pd.DataFrame(col_colors, index=labels_pred_sorted)

    xticklabels = predicted_ticklabels if predicted_ticklabels is not None else (True if len(labels_pred) <= 40 else False)
    yticklabels = true_ticklabels if true_ticklabels is not None else (True if len(labels_true) <= 40 else False)
    annot = annot if annot is not None else (True if len(labels_true) <= 40 and len(labels_pred) <= 40 else False)


    g = sns.clustermap(cm_normalized, annot=annot, fmt=".2f", cmap="Blues",
                       row_colors=row_colors, col_colors=col_colors,
                       xticklabels=xticklabels, yticklabels=yticklabels,
                       row_cluster=False, col_cluster=False, figsize=figsize)

    g.ax_heatmap.set_title(title, y=1.05)
    g.ax_heatmap.set_ylabel('True label')
    g.ax_heatmap.set_xlabel('Predicted label')
    plt.show()

    return g

def plot_sankey(adata, cols, params=None):
    """
    Generate a Sankey diagram from the specified columns in the `adata.obs` DataFrame.

    Parameters:
        adata : AnnData
            An AnnData object containing the observation data (`adata.obs`) to be visualized.
        cols : list of str
            A list of column names from `adata.obs` that define the nodes and flow relationships for the Sankey diagram.
        params : dict, optional
            A dictionary of optional parameters to customize the Sankey diagram appearance. Supported keys include:
                - 'cmap': str, colormap for node colors (default: 'Colorblind').
                - 'label_position': str, position of node labels ('outer' or 'center', default: 'outer').
                - 'edge_line_width': int, width of the edges (default: 0).
                - 'edge_color': str, attribute for edge coloring (default: 'value', or 'grey' for uniform color).
                - 'show_values': bool, whether to display flow values (default: False).
                - 'node_padding': int, padding between nodes (default: 12).
                - 'node_alpha': float, transparency of nodes (default: 1.0).
                - 'node_width': int, width of nodes (default: 30).
                - 'node_sort': bool, whether to sort nodes (default: True).
                - 'frame_height': int, height of the diagram frame (default: 1000).
                - 'frame_width': int, width of the diagram frame (default: 2000).
                - 'bgcolor': str, background color of the diagram (default: 'white').
                - 'apply_ranges': bool, whether to apply range adjustments to the plot (default: True).
                - 'align_thr': float, alignment threshold for colors (default: -0.1).
                - 'label_font_size': str, font size for labels (default: '12pt').

    Returns:
        hv.Sankey
            A Holoviews Sankey diagram object configured based on the input data and parameters.

    Example:
        sankey = plot_sankey(adata, cols=['column1', 'column2', 'column3'], params={'cmap': 'viridis', 'frame_width': 1200})
        hv.save(sankey, 'sankey_diagram.html')
    """
    import holoviews as hv
    hv.extension('bokeh')

    def f(plot, element):
        plot.handles['plot'].sizing_mode = 'scale_width'
        plot.handles['plot'].x_range.start = -1000
        plot.handles['plot'].x_range.end = 1500


    if params is None:
        params = {}

    obs = adata.obs[cols]

    # Creating unique labels for each column
    unique_labels = []
    label_dict = defaultdict(dict)
    for col_index, col in enumerate(cols):
        col_data = obs[col].astype(str).tolist()
        for item in col_data:
            if item not in label_dict[col_index]:
                unique_label = f"{item} ({col})"
                label_dict[col_index][item] = unique_label
                unique_labels.append(unique_label)

    # Creating source, target and value lists
    source = []
    target = []
    value = []
    for i in range(len(cols) - 1):
        ct_dict = defaultdict(int)
        for a, b in zip(obs[cols[i]].astype(str), obs[cols[i+1]].astype(str)):
            ct_dict[(a, b)] += 1
        for (a, b), v in ct_dict.items():
            source.append(label_dict[i][a])
            target.append(label_dict[i+1][b])
            value.append(v)

    # Creating DataFrame for Sankey
    sankey_data = pd.DataFrame({
        'source': source,
        'target': target,
        'value': value
    })

    # Appearance parameters
    cmap = params.get('cmap', 'Colorblind')
    label_position = params.get('label_position', 'outer')
    edge_line_width = params.get('edge_line_width', 0)
    edge_color = params.get('edge_color', 'value')  # allows grey edges
    show_values = params.get('show_values', False)
    node_padding = params.get('node_padding', 12)
    node_alpha = params.get('node_alpha', 1.0)
    node_width = params.get('node_width', 30)
    node_sort = params.get('node_sort', True)
    frame_height = params.get('frame_height', 1000)
    frame_width = params.get('frame_width', 2000)
    bgcolor = params.get('bgcolor', 'white')
    apply_ranges = params.get('apply_ranges', True)
    align_thr = params.get('align_thr', -0.1)
    label_font_size = params.get('label_font_size', '12pt')

    colormap_max = max(sankey_data['value'])
    norm = plt.Normalize(vmin=0, vmax=colormap_max)
    colors = plt.cm.get_cmap("plasma")(norm(np.linspace(0, colormap_max, 128)))

    replace_these = np.where(norm(np.linspace(0, colormap_max, 128)) <= align_thr)[0]
    if replace_these.size > 0:
        colors[replace_these] = [[1, 1, 1, 0] for _ in range(len(replace_these))]

    edge_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    if edge_color == "grey":
        # edge_color = "grey"  # Set edge_color to grey
        edge_cmap = None  # No colormap for grey edges

    sankey = hv.Sankey(sankey_data, kdims=["source", "target"], vdims=["value"])
    sankey = sankey.opts(
        cmap=cmap, label_position=label_position, edge_color=edge_color, edge_cmap=edge_cmap, colorbar=True if edge_cmap else False,
        edge_line_width=edge_line_width, show_values=show_values, node_padding=node_padding, node_alpha=node_alpha,
        node_width=node_width, node_sort=node_sort, frame_height=frame_height, frame_width=frame_width,
        bgcolor=bgcolor, apply_ranges=apply_ranges, label_text_font_size=label_font_size, hooks=[f]
    )
    sankey = sankey.opts(clim=(0, colormap_max))

    return sankey

def save_sankey(plot, filename, adt_key=None):
    """
    Save a Holoviews Sankey plot as an SVG file.

    Parameters:
    plot : Holoviews plot, The Sankey plot to save.
    filename : str Base filename for the output SVG file.
    adt_key : str, optional Optional identifier to append to the filename.
    """
    import holoviews as hv
    from bokeh.io.webdriver import webdriver_control
    from bokeh.io import export_svgs

    # Reset web driver because sometimes the max connections is hit when writing plots
    webdriver_control.reset()

    # Remove '.svg' if it exists and append '{adt_key}.svg'
    filename = os.path.splitext(filename)[0]
    if adt_key:
        filename += f"_{adt_key}"
    filename += ".svg"

    plot = hv.render(plot)
    plot.output_backend = "svg"

    export_svgs(plot, filename=filename)


def plot_grouped_average(adata, label_value, adt_key=None):
    """
    Plots the average values specified in label_value across each group of label_keys in an AnnData object.

    Parameters:
    - adata: AnnData object containing the data.
    - label_value: dict, keys are the keys in adata.obs for grouping, values are the keys in adata.obs for the values to average.
    - key: to print specified key
    """
    print(adt_key)
    if not all(label in adata.obs for label in label_value.keys()):
        missing_keys = [label for label in label_value.keys() if label not in adata.obs]
        raise ValueError(f"Label key(s) {missing_keys} not found in adata.obs.")
    if not all(value in adata.obs for value in label_value.values()):
        missing_values = [value for value in label_value.values() if value not in adata.obs]
        raise ValueError(f"Value key(s) {missing_values} not found in adata.obs.")
    
    grouped_means = {}
    for label, value in label_value.items():
        grouped_means[label] = adata.obs.groupby(label)[value].mean()

    # Create a DataFrame from the grouped means
    df = pd.DataFrame(grouped_means)
    
    # Plot the results
    df.plot(kind='bar', figsize=(12, 8), color=plt.cm.get_cmap("Paired").colors)
    plt.xlabel('Groups')
    plt.ylabel('Average Scores')
    plt.title('Average Scores across Groups')
    plt.xticks(rotation=90)
    plt.legend(title='Scores')
    plt.show()


def plot_model_agreement(adata, group_by, sub_group_by, model_cols, granularity=2):
    """
    Plots the average values of specified model columns across varying levels of granularity.
    Parameters:
    - adata: AnnData object containing the data.
    - group_by: str, key in adata.obs for the main grouping (e.g., 'cell_type').
    - sub_group_by: str, key in adata.obs for the sub-grouping (e.g., 'tissue').
    - model_cols: list of str, column names for the models (e.g., ['agreement_model_1', 'agreement_model_2']).
    - granularity: int, level of detail in the plot (0 = models only, 1 = models within cell types, 2 = models within cell types and tissues).
    """
    if not all(col in adata.obs for col in model_cols):
        missing_cols = [col for col in model_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Pivot longer to get columns: group_by, sub_group_by, agreement, model_name
    melted = adata.obs.melt(id_vars=[group_by, sub_group_by], value_vars=model_cols,
                            var_name='model_name', value_name='agreement')

    if granularity == 0:
        # Calculate the average scores across all groups within each model
        grouped_means = melted.groupby('model_name')['agreement'].mean().sort_values(ascending=False)

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the bar chart
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')

        # Add value labels on top of each bar
        for i, v in enumerate(grouped_means):
            ax.text(i, v, f'{v * 100:.0f}%', ha='center', va='bottom')

    elif granularity == 1:
        # Calculate the average scores within each model and cell type
        grouped_means = melted.groupby([group_by, 'model_name'])['agreement'].mean().unstack()

        fig, ax = plt.subplots(figsize=(14, 8))
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')

    elif granularity == 2:
        # Calculate the average scores within each model, cell type, and tissue
        grouped_means = melted.groupby([group_by, sub_group_by, 'model_name'])['agreement'].mean().unstack(level=[1, 2])

        # Ensure the data is numeric and allow NaNs (missing values)
        grouped_means = grouped_means.apply(pd.to_numeric, errors='coerce')

        # Create a mask for NaN values
        mask = grouped_means.isnull()

        # Create a color mapping for tissues using the provided colors
        tissue_colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
            "#7f9ec0", "#ffab60", "#5ab4ac",
            # 5 additional concordant colors
            "#8b4513", "#ff6347", "#4682b4", "#dda0dd", "#ffd700"
        ]

        # Ensure that the number of tissues does not exceed the number of available colors
        tissues = grouped_means.columns.get_level_values(0).unique()
        tissue_colors = tissue_colors[:len(tissues)]

        # Create a color map based on the provided colors
        tissue_color_map = dict(zip(tissues, tissue_colors))

        # Create column colors based on tissues
        col_colors = [tissue_color_map[tissue] for tissue in grouped_means.columns.get_level_values(0)]

        
        # Plot heatmap with col_colors
        # fig, ax = plt.subplots(figsize=(16, 10))
        # Create the clustermapimport seaborn as sns

        # Use the 'viridis_r' colormap
        cmap = plt.get_cmap('viridis_r')

        # Set the color for NaN values (e.g., red)
        cmap.set_bad(color='black')

        # Create the clustermap with horizontal lines
        g = sns.clustermap(grouped_means, cmap=cmap, annot=False,
                        mask=mask, cbar_kws={'label': 'Agreement'},
                        linewidths=0, linecolor='black',
                        col_colors=col_colors, col_cluster=False, row_cluster=False,
                        yticklabels=1)

        # Get the axes object
        ax = g.ax_heatmap

        # # Remove all existing lines
        # ax.grid(False)

        # Add back only horizontal lines
        # ax.set_xticks(np.arange(grouped_means.shape[1]+1)-0.5, minor=True)
        # ax.set_yticks(np.arange(grouped_means.shape[0]+1)-0.5, minor=True)
        # ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        # ax.tick_params(which="minor", bottom=False, left=False)

        # Find where col_colors change
        color_changes = []
        for i in range(1, len(col_colors)):
            if col_colors[i] != col_colors[i-1]:
                color_changes.append(i)

        # Add vertical lines at color change positions
        for pos in color_changes:
            ax.axvline(pos, color='black', linewidth=0.5)

        return g

        # Create a legend for tissues
        # tissue_handles = [plt.Rectangle((0,0),1,1, color=color) for color in tissue_color_map.values()]
        # ax.legend(tissue_handles, tissue_color_map.keys(), title=sub_group_by, 
        #           loc='center left', bbox_to_anchor=(1, 0.5))
        # return fig, ax

    else:
        raise ValueError("Granularity must be 0, 1, or 2.")

    if granularity < 2:
        ax = plt.gca()  # Get current axis for granularity 0 and 1

    ax.set_xlabel(group_by if granularity > 0 else 'Model')
    ax.set_ylabel('Agreement')
    title = 'Average model agreement'
    if granularity == 0:
        title += ''
    elif granularity == 1:
        title += f' by {group_by}'
    elif granularity == 2:
        title += f' by {group_by} and {sub_group_by}'
    ax.set_title(title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

    if granularity < 2:
        ax.legend(title='Models' + ('' if granularity == 0 else ' and Tissues'))

    plt.tight_layout()
    # Return the figure and axis for further editing
    return fig, ax

def plot_model_agreement_categorical(adata, group_by, sub_group_by, model_cols, granularity=2):
    """
    Plots the relative proportions of categories within specified model columns across varying levels of granularity.

    Parameters:
    - adata: AnnData object containing the data.
    - group_by: str, key in adata.obs for the main grouping (e.g., 'cell_type').
    - sub_group_by: str, key in adata.obs for the sub-grouping (e.g., 'tissue').
    - model_cols: list of str, column names for the models (e.g., ['model_1', 'model_2']). These should be categorical.
    - granularity: int, level of detail in the plot (0 = models only, 1 = models within cell types, 2 = models within cell types and tissues).
    """
    # Verify that the required columns exist
    if not all(col in adata.obs for col in model_cols):
        missing_cols = [col for col in model_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Ensure that model_cols are categorical or convert numeric types to categories
    for col in model_cols:
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype('category')
            else:
                raise ValueError(f"Column '{col}' must be categorical or convertible to categorical.")

    # Melt the dataframe to get long format
    melted = adata.obs.melt(
        id_vars=[group_by, sub_group_by],
        value_vars=model_cols,
        var_name='model_name',
        value_name='agreement'
    )

    # Ensure 'agreement' is categorical and reverse the order of categories
    if not pd.api.types.is_categorical_dtype(melted['agreement']):
        melted['agreement'] = melted['agreement'].astype('category')

    # Reverse the order of 'agreement' categories
    original_categories = melted['agreement'].cat.categories.tolist()
    reversed_categories = original_categories[::-1]
    melted['agreement'] = melted['agreement'].cat.reorder_categories(reversed_categories, ordered=True)

    if granularity == 0:
        # Calculate counts and proportions
        counts = melted.groupby(['model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby('model_name')['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Sort models based on total proportion of highest agreement category
        highest_agreement = counts.groupby('model_name')['proportion'].max().reset_index()
        sorted_models = highest_agreement.sort_values('proportion', ascending=False)['model_name']
        counts['model_name'] = pd.Categorical(counts['model_name'], categories=sorted_models, ordered=True)

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=counts,
            x='model_name',
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            ax=ax,
            order=sorted_models
        )

        # Add proportion labels on top of each bar
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height + 0.01,
                    # f'{height:.2f}',
                    f'{height * 100:.0f}%',
                    ha="center"
                )

        # Rotate x-axis tick labels to vertical
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel('Model')
        ax.set_ylabel('Proportion')
        ax.set_title('Proportion of Agreement Categories by Model')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Agreement Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig, ax

    elif granularity == 1:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Sort 'group_by' categories based on total proportion
        total_per_group = counts.groupby(group_by)['proportion'].sum().reset_index()
        sorted_groups = total_per_group.sort_values('proportion', ascending=False)[group_by]
        counts[group_by] = pd.Categorical(counts[group_by], categories=sorted_groups, ordered=True)

        # Plot grouped bar chart with model_name as hue
        g = sns.catplot(
            data=counts,
            x=group_by,
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            col='model_name',
            kind='bar',
            height=6,
            aspect=1,
            sharey=True,
            order=sorted_groups
        )

        g.set_axis_labels(group_by, "Proportion")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1.05))

        # Rotate x-axis tick labels to vertical
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Add proportion labels on top of each bar
        for ax in g.axes.flatten():
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        height + 0.01,
                        # f'{height:.2f}',
                        f'{height * 100:.0f}%',
                        ha="center"
                    )

        plt.tight_layout()
        return g

    elif granularity == 2:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, sub_group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, sub_group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Prepare data for heatmap
        pivot_table = counts.pivot_table(
            index=[group_by, sub_group_by],
            columns=['model_name', 'agreement'],
            values='proportion',
            fill_value=0
        )

        # Reverse the order of 'agreement' categories in columns
        pivot_table = pivot_table.reindex(columns=reversed_categories, level=2)

        # Sort index based on total proportion
        pivot_table['Total'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values('Total', ascending=False)
        pivot_table = pivot_table.drop(columns='Total')

        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_table,
            cmap='viridis',
            annot=True,
            fmt=".2f",
            linewidths=0.5
        )
        plt.title(f'Proportion of Agreement Categories by {group_by} and {sub_group_by}')
        plt.tight_layout()
        return plt.gcf()

    else:
        raise ValueError("Granularity must be 0, 1, or 2.")
