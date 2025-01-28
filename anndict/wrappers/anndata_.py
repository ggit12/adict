"""
This module contains and adata_dict wrappers for `anndata`.
Some of these ease manipulation and information retrieval from andata.
"""
import numpy as np

from anndict.adata_dict import adata_dict_fapply, adata_dict_fapply_return
from anndict.utils import enforce_semantic_list

def set_var_index(adata_dict, column):
    """
    Set the index of adata.var to the specified column for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary where keys are identifiers and values are AnnData objects.
    column (str): The column name to set as the index of adata.var.

    Returns:
    dict: A dictionary with the same structure as adata_dict, where the var attribute of each AnnData object has its index set to the specified column.
    """
    def set_var_index_main(adata, column):
        adata.var = adata.var.set_index(column)
        return adata
    return adata_dict_fapply_return(adata_dict, set_var_index_main, column=column)


def set_obs_index(adata_dict, column):
    """
    Set the index of adata.obs to the specified column for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary where keys are identifiers and values are AnnData objects.
    column (str): The column name to set as the index of adata.obs.

    Returns:
    dict: A dictionary with the same structure as adata_dict, where the obs attribute of each AnnData object has its index set to the specified column.
    """
    def set_obs_index_main(adata, column):
        adata.obs = adata.obs.set_index(column)
        return adata

    return adata_dict_fapply_return(adata_dict, set_obs_index_main, column=column)

def remove_genes(adata, genes_to_remove, adt_key=None):
    """
    Remove specified genes from an AnnData object in-place.

    Parameters:
    adata (anndata.AnnData): The AnnData object to modify.
    genes_to_remove (list): A list of gene names to remove.

    Returns:
    None
    """
    # Get the list of genes to remove that are actually in the dataset
    genes_to_remove = adata.var_names.intersection(genes_to_remove)

    # Remove the specified genes
    # (the only way to do this in-place for now is to use the protected member of the class)
    adata._inplace_subset_var(~adata.var_names.isin(genes_to_remove))

    print(f"Removed {len(genes_to_remove)} genes from {adt_key}. {adata.n_vars} genes remaining.")

def remove_genes_adata_dict(adata_dict, genes_to_remove):
    """
    Remove specified genes from each AnnData object in adata_dict.

    Parameters:
    adata_dict : dict A dictionary where keys are identifiers and values are AnnData objects.
    genes_to_remove : list A list of gene names to remove from each AnnData object.

    Returns:
    None
    """
    adata_dict_fapply(adata_dict, remove_genes, genes_to_remove=genes_to_remove)


def add_label_to_adata(adata, indices, labels, new_label_key):
    """
    Adds a label to the AnnData object in a specified column for given indices.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - labels: Array of labels corresponding to the indices.
    - new_label_key: Name of the column in adata.obs where the labels will be stored.
    """
    add_col_to_adata_obs(adata, indices, labels, new_label_key)
    

def add_col_to_adata_obs(adata, indices, values, new_col_name):
    """
    Adds a label to the AnnData object in a specified column for given indices.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - values: Array of labels corresponding to the indices.
    - new_col_name: Name of the column in adata.obs where the labels will be stored.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.obs[new_col_name] = np.full(adata.obs.shape[0], np.nan, dtype=dtype)
    adata.obs.loc[indices, new_col_name] = values


def add_col_to_adata_var(adata, indices, values, new_col_name):
    """
    Adds a label to the AnnData object in a specified column for given indices in adata.var.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - values: Array of labels corresponding to the indices.
    - new_label_key: Name of the column in adata.var where the labels will be stored.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.var[new_col_name] = np.full(adata.var.shape[0], np.nan, dtype=dtype)
    adata.var.loc[indices, new_col_name] = values


def convert_obs_col_to_category(adata, col_name):
    """
    Convert a column in AnnData.obs to category dtype.
    
    Parameters:
    adata (AnnData): The AnnData object.
    col_name (str): The name of the column in adata.obs to convert.
    
    Returns:
    None: The function modifies the adata object in-place.
    """
    if col_name not in adata.obs.columns:
        raise ValueError(f"Column '{col_name}' not found in adata.obs")
    
    adata.obs[col_name] = adata.obs[col_name].astype('category')


def convert_obs_col_to_string(adata, col_name):
    """
    Convert a column in AnnData.obs to string dtype.
    
    Parameters:
    adata (AnnData): The AnnData object.
    col_name (str): The name of the column in adata.obs to convert.
    
    Returns:
    None: The function modifies the adata object in-place.
    """
    if col_name not in adata.obs.columns:
        raise ValueError(f"Column '{col_name}' not found in adata.obs")
    
    adata.obs[col_name] = adata.obs[col_name].astype(str)


def convert_obs_index_to_str(adata):
    """
    Converts the index of .obs to a string
    """
    adata.obs.index = adata.obs.index.astype(str)


def get_adata_columns(adata, col_startswith=None, col_endswith=None, col_contains=None, 
                      not_col_startswith=None, not_col_endswith=None, not_col_contains=None):
    """
    Extract columns from an AnnData object's observation dataframe (`adata.obs`) based on specified filtering criteria.

    Parameters:
    adata : AnnData The AnnData object containing the observation dataframe (adata.obs) from which columns are selected.
    col_{startswith, endswith, contains} : list of str, optional
    Lists of substrings to positively filter columns. 
    - col_startswith: Select columns that start with any of these strings.
    - col_endswith: Select columns that end with any of these strings.
    - col_contains: Select columns that contain any of these strings.
    not_col_{startswith, endswith, contains} : list of str, optional
    Lists of substrings to negatively filter columns from the previously selected ones. 
    - not_col_startswith: Exclude columns that start with any of these strings.
    - not_col_endswith: Exclude columns that end with any of these strings.
    - not_col_contains: Exclude columns that contain any of these strings.

    Returns:
    list of str A list of unique column names from adata.obs that match the specified criteria.
    """
    columns = adata.obs.columns
    matched_columns = []

    #Get local variables and enforce that the filtering parameters are lists
    filter_params = ['col_startswith', 'col_endswith', 'col_contains', 
                     'not_col_startswith', 'not_col_endswith', 'not_col_contains']

    for param in filter_params:
        val = locals()[param]
        if isinstance(val, str):
            locals()[param] = [val]

    if col_startswith:
        for start in col_startswith:
            matched_columns.extend([col for col in columns if col.startswith(start)])

    if col_endswith:
        for end in col_endswith:
            matched_columns.extend([col for col in columns if col.endswith(end)])

    if col_contains:
        for contain in col_contains:
            matched_columns.extend([col for col in columns if contain in col])

    if not_col_startswith:
        for start in not_col_startswith:
            matched_columns = [col for col in matched_columns if not col.startswith(start)]

    if not_col_endswith:
        for end in not_col_endswith:
            matched_columns = [col for col in matched_columns if not col.endswith(end)]

    if not_col_contains:
        for contain in not_col_contains:
            matched_columns = [col for col in matched_columns if contain not in col]

    return list(set(matched_columns))


def filter_gene_list(adata, gene_list):
    """
    Filter and update a list of gene names based on their presence in the index of an AnnData object.

    Parameters:
        adata : AnnData
            AnnData object containing gene information in `adata.var.index`.
        gene_list : list of str
            List of gene names to be filtered and updated with possible unique suffixes.

    Returns:
        list of str
            Updated list of genes found in `adata.var.index`, including suffix variations.
    """
    enforce_semantic_list(adata.var.index)
    updated_gene_list = []
    for gene in gene_list:
        # Create a regex pattern to match the gene name and its possible unique suffixes, case-insensitive
        pattern = re.compile(r'^' + re.escape(gene) + r'(-\d+)?$', re.IGNORECASE)

        # Find all matching genes in adata.var.index
        matching_genes = [g for g in adata.var.index if pattern.match(g)]

        if matching_genes:
            updated_gene_list.extend(matching_genes)
        # else:
            # print(f"Gene '{gene}' not found in adata.var.index after making unique.")

    # Remove any duplicates in the updated marker list
    updated_gene_list = list(set(updated_gene_list))
    return updated_gene_list


def summarize_metadata(adata, columns):
    """
    Generate a summary for specified metadata columns in an anndata object as a dictionary.
    
    Parameters:
        adata (anndata.AnnData): The anndata object containing the data.
        columns (list of str): List of columns from the metadata. 
                               Use '*' to specify joint frequencies of multiple columns.
    
    Returns:
        dict: A dictionary with keys as column descriptions and values as a DataFrame of counts.
    """
    results = {}
    
    for col in columns:
        if '*' in col:
            # Handle joint frequencies
            sub_cols = col.split('*')
            combined_data = adata.obs[sub_cols]

            # Convert categorical columns to string and replace NaN with a placeholder
            for sub_col in sub_cols:
                if pd.api.types.is_categorical_dtype(combined_data[sub_col]):
                    combined_data[sub_col] = combined_data[sub_col].astype(str)

            # Replace NaN with a placeholder to include them in groupby
            combined_data = combined_data.fillna('NaN')

            joint_freq = combined_data.groupby(sub_cols).size().unstack(fill_value=0)
            joint_freq = combined_data.groupby(sub_cols, observed=True).size().unstack(fill_value=0)
            results[col.replace('*', ' x ')] = joint_freq
        else:
            # Calculate frequency for a single column
            freq = adata.obs[col].value_counts(dropna=False).to_frame('count')
            results[col] = freq
    
    return results



