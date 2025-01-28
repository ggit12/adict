"""
This module contains functions that process single columns of annotations.
"""

import pandas as pd

from anndict.utils import make_names, enforce_semantic_list
from anndict.llm import retry_call_llm, extract_dictionary_from_ai_string, process_llm_category_mapping
from anndict.wrappers import convert_obs_col_to_category

def simplify_obs_column(adata, column, new_column_name, simplification_level=''):
    """
    Simplifies labels in the specified column of the AnnData object and stores the result
    in a new column using the map_cell_type_labels_to_simplified_set().

    Args:
    adata (AnnData): The AnnData object containing the data.
    column (str): The column in adata.obs containing the cell type labels to simplify.
    new_column_name (str): The name of the new column to store the simplified labels.
    simplification_level (str, optional): Defaults to ''. A qualitative description of how much you want the labels to be simplified. Could be anything, like  'extremely', 'barely', or 'compartment-level'.

    """
    # Get the unique labels from the specified column
    unique_labels = adata.obs[column].unique()

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_cell_type_labels_to_simplified_set(unique_labels, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.obs[new_column_name] = adata.obs[column].map(label_mapping)

    #Convert annotation to categorical dtype
    convert_obs_col_to_category(adata, new_column_name)

    return label_mapping

def simplify_var_index(adata, column, new_column_name, simplification_level=''):
    """
    Simplifies gene names in the index of the AnnData object's var attribute based on a boolean column,
    and stores the result in a new column using the map_gene_labels_to_simplified_set().

    Args:
    adata (AnnData): The AnnData object containing the data.
    column (str): The boolean column in adata.var used to select genes for simplification.
    new_column_name (str): The name of the new column to store the simplified labels.
    simplification_level (str, optional): Defaults to ''. A qualitative description of how much you want the labels to be simplified. Could be anything, like 'extremely', 'barely', or 'compartment-level'.

    Raises:
    ValueError: If more than 1000 genes are selected for simplification or if the column is not boolean.
    """
    if not pd.api.types.is_bool_dtype(adata.var[column]):
        raise ValueError(f"The column '{column}' must be a boolean index column.")

    # Get the index of the true indices in the boolean column
    selected_genes = adata.var.index[adata.var[column]]

    if len(selected_genes) > 1000:
        raise ValueError("Cannot simplify more than 1000 genes at a time.")

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_gene_labels_to_simplified_set(selected_genes, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.var[new_column_name] = adata.var.index.to_series().map(label_mapping).fillna(adata.var.index.to_series())

    return label_mapping


def create_label_hierarchy(adata, col, simplification_levels):
    """
    Create a hierarchy of simplified labels based on a given column in AnnData.

    This function generates multiple levels of simplified labels from an original
    column in the AnnData object. Each level of simplification is created using
    the specified simplification levels.

    Parameters:
    adata : AnnData Annotated data matrix containing the column to be simplified.
    col : str Name of the column in adata.obs to be simplified.
    simplification_levels : list List of simplification levels to apply. Each level should be a value that can be used by the simplify_obs_column function.

    Returns:
    --------
    dict A dictionary mapping new column names to their corresponding simplified label mappings. The keys are the names of the new columns created for each simplification level, and the values are the mappings returned by simplify_obs_column for each level.
    """
    base_col_name = col
    simplified_mapping = {}
    for level in simplification_levels:
        new_col_name = f"{base_col_name}_{make_names([level])[0]}"
        simplified_mapping[new_col_name] = simplify_obs_column(adata, col, new_col_name, simplification_level=level)
        col = new_col_name
    return simplified_mapping


#Label simplification functions
def map_cell_type_labels_to_simplified_set(labels, simplification_level='', batch_size=50):
    """
    Maps a list of labels to a smaller set of labels using the AI, processing in batches.
    Args:
    labels (list of str): The list of labels to be mapped.
    simplification_level (str): A qualitative description of how much you want the labels to be simplified. Or a direction about how to simplify the labels. Could be anything, like 'extremely', 'barely', 'compartment-level', 'remove-typos'
    batch_size (int): The number of labels to process in each batch.
    Returns:
    dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    #todo, could allow passing custom examples
    #enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)
    
    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generator that takes a list of categories and provides a mapping to a {simplification_level} simplified set as a dictionary. Generate only a dictionary. Example: Fibroblast.    Fibroblasts.    CD8-positive T Cells.    CD4-positive T Cells. -> {{'Fibroblast.':'Fibroblast','Fibroblasts.':'Fibroblast','CD8-positive T Cells.':'T Cell','CD4-positive T Cells.':'T Cell'}}"},
        {"role": "user", "content": f"Here is the full list of labels to be simplified: {initial_labels_str}. Acknowledge that you've seen all labels. Do not provide the mapping yet."}
    ]

    # Get initial acknowledgment
    initial_response = retry_call_llm(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to process initial prompt",
        call_llm_kwargs={'max_tokens': 30, 'temperature': 0},
        max_attempts=1
    )
    messages.append({"role": "assistant", "content": initial_response})

    def process_batch(batch_labels):
        batch_str = "    ".join(batch_labels)
        messages.append({"role": "user", "content": f"Provide a mapping for this batch of labels. Generate only a dictionary: {batch_str} -> "})
        
        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for labels: {labels}")
            return {label: label for label in labels}

        call_llm_kwargs = {
            'max_tokens': min(300 + 25*len(batch_labels), 4000),
            'temperature': 0
        }
        failure_handler_kwargs = {'labels': batch_labels}

        batch_mapping = retry_call_llm(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            call_llm_kwargs=call_llm_kwargs,
            failure_handler_kwargs=failure_handler_kwargs
        )
        messages.append({"role": "assistant", "content": str(batch_mapping)})
        return batch_mapping

    # Process all labels in batches
    full_mapping = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_mapping = process_batch(batch)
        full_mapping.update(batch_mapping)

    # Final pass to ensure consistency
    final_mapping = process_llm_category_mapping(labels, full_mapping)
    
    return final_mapping

def map_gene_labels_to_simplified_set(labels, simplification_level='', batch_size=50):
    """
    Maps a list of genes to a smaller set of labels using AI, processing in batches.
    Args:
    labels (list of str): The list of labels to be mapped.
    simplification_level (str): A qualitative description of how much you want the labels to be simplified.
    batch_size (int): The number of labels to process in each batch.
    Returns:
    dict: A dictionary mapping the original labels to the smaller set of labels.
    """
    # Enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)
    
    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": f"You are a python dictionary mapping generator that takes a list of genes and provides a mapping to a {simplification_level} simplified set as a dictionary. Example: HSP90AA1    HSPA1A    HSPA1B    CLOCK    ARNTL    PER1    IL1A    IL6 -> {{'HSP90AA1':'Heat Shock Proteins','HSPA1A':'Heat Shock Proteins','HSPA1B':'Heat Shock Proteins','CLOCK':'Circadian Rhythm','ARNTL':'Circadian Rhythm','PER1':'Circadian Rhythm','IL1A':'Interleukins','IL6':'Interleukins'}}"},
        {"role": "user", "content": f"Here is the full list of gene labels to be simplified: {initial_labels_str}. Acknowledge that you've seen all labels. Do not provide the mapping yet."}
    ]

    # Get initial acknowledgment
    initial_response = retry_call_llm(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to process initial prompt",
        call_llm_kwargs={'max_tokens': 30, 'temperature': 0},
        max_attempts=1
    )
    messages.append({"role": "assistant", "content": initial_response})

    def process_batch(batch_labels):
        batch_str = "    ".join(batch_labels)
        messages.append({"role": "user", "content": f"Provide a mapping for this batch of gene labels. Generate only a dictionary: {batch_str} -> "})
        
        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for gene labels: {labels}")
            return {label: label for label in labels}

        call_llm_kwargs = {
            'max_tokens': min(300 + 25*len(batch_labels), 4000),
            'temperature': 0
        }
        failure_handler_kwargs = {'labels': batch_labels}

        batch_mapping = retry_call_llm(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            call_llm_kwargs=call_llm_kwargs,
            failure_handler_kwargs=failure_handler_kwargs
        )
        messages.append({"role": "assistant", "content": str(batch_mapping)})
        return batch_mapping

    # Process all labels in batches
    full_mapping = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_mapping = process_batch(batch)
        full_mapping.update(batch_mapping)

    # Final pass to ensure consistency
    final_mapping = process_llm_category_mapping(labels, full_mapping)
    
    return final_mapping