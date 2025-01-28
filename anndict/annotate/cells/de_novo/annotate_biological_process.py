"""
This module annotates groups of cells with a biological process, based on the group's enriched genes
"""

from anndict.utils import enforce_semantic_list
from anndict.llm import call_llm
from anndict.annotate.cells.de_novo.base import ai_annotate


def ai_biological_process(gene_list):
    """
    Describes the most prominent biological process represented by a list of genes using an LLM.

    Args:
        gene_list (list of str): The list of genes to be described.

    Returns:
        dict: A dictionary containing the description of the biological process.
    """
    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        base_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list[:-1])
        base_prompt = f"In a few words and without restating any part of the question, describe the single most prominent biological process represented by the genes: {genes_str}, and {gene_list[-1]}"

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": base_prompt}
    ]

    # Call the LLM using the call_llm function
    annotation = call_llm(
        messages=messages,
        max_tokens=200,
        temperature=0
    )

    return annotation


def ai_annotate_biological_process(adata, groupby, n_top_genes, label_column='ai_biological_process'):
    """
    Annotate biological processes based on the top n marker genes for each cluster.

    This function performs differential expression analysis to identify marker genes for each cluster
    and applies a user-defined function to determine the biological processes for each cluster based on the top 
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    Parameters:
    adata : AnnData
    groupby : str Column in adata.obs to group by for differential expression analysis.
    n_top_genes : int The number of top marker genes to consider for each cluster.
    label_column : str, optional (default: 'ai_cell_type') The name of the new column in adata.obs where the cell type annotations will be stored.

    Returns:
    pd.DataFrame A DataFrame with a column for the top marker genes for each cluster.
    """
    return ai_annotate(func=ai_biological_process, adata=adata, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


