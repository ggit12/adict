"""
This module contains functions that calculate cell type marker gene scores automaticallly (i.e. you supply only the cell type, not the marker genes).
"""
import numpy as np
import scanpy as sc

from anndict.wrappers.anndata_ import filter_gene_list
from anndict.annotate.genes import ai_gene_list


def cell_type_marker_gene_score(adata, cell_type_col=None, cell_types=None, species='Human', list_length=None, score_name='_score', adt_key=None, **kwargs):
    """
    Compute marker gene scores for specified cell types. Must provide either a list of cell types, or a column that contains cell_type labels.

    Parameters:
        adata (AnnData): Annotated data matrix.
        cell_type_col (str, optional): Column name in adata.obs containing cell type annotations.
        cell_types (list of str, optional): List of cell types for which to compute the marker gene scores.
        species (str, optional): Species for gene list generation. Defaults to 'Human'.
        list_length (str, optional): Qualitative length of the marker gene list (i.e. "longer" if you are having trouble getting valid genes present in your dataset.)
        score_name (str, optional): Suffix for the computed score names. Defaults to '_score'.
        **kwargs: Optional keyword args passed to sc.tl.score_genes().

    Modifies:
        adata.var: Adds boolean columns indicating genes used in the scores.
        adata.obs: Adds new columns with the computed scores for each observation.

    """

    score_name_suffix = score_name

    # Check for conflicting parameters
    if cell_types is not None and cell_type_col is not None:
        raise ValueError("Provide either 'cell_type_col' or 'cell_types', not both.")

    if cell_types is None:
        if cell_type_col is not None:
            cell_types = adata.obs[cell_type_col].unique().tolist()
        else:
            raise ValueError("Either 'cell_type_col' or 'cell_types' must be provided.")
    else:
        # Ensure cell_types is a list
        if isinstance(cell_types, str):
            cell_types = [cell_types]

    for cell_type in cell_types:
        cell_type = str(cell_type)  # Ensure cell_type is a string
        # Set the score_name per cell type
        score_name = f"{cell_type}{score_name_suffix}"

        # Generate gene list using ai_gene_list function
        gene_list = ai_gene_list(cell_type, species, list_length=list_length)

        # Filter the gene list based on genes present in adata
        gene_list = filter_gene_list(adata, gene_list)

        # Mark genes included in this score in adata.var
        adata.var[score_name] = adata.var.index.isin(gene_list)

        #calculate score if any valid genes, otherwise print warning and assign score value as NaN.
        if gene_list:
            # Compute the gene score and store it in adata.obs[score_name]
            sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, **kwargs)
        else:
            # Assign NaN to adata.obs[score_name] for all observations
            adata.obs[score_name] = np.nan
            print(f"No valid genes for {cell_type} in {adt_key if adt_key else ''}. Assigning score value as NaN")
