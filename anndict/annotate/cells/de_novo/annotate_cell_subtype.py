"""
This module contains functions to annotate cell subtype
"""
from anndict.adata_dict import build_adata_dict, concatenate_adata_dict
from anndict.wrappers.anndictionary_ import ai_annotate_cell_type_by_comparison_adata_dict

def ai_annotate_cell_sub_type(adata, cell_type_col, sub_cluster_col, new_label_col, tissue_of_origin_col=None, n_top_genes=10):
    """
    Annotate cell subtypes using AI.

    This function performs AI-based annotation of cell subtypes by first grouping cells
    by their main cell type, then annotating subtypes within each group.

    Parameters:
    adata : AnnData Annotated data matrix.
    cell_type_col : str Column name in adata.obs containing main cell type labels.
    sub_cluster_col : str Column name in adata.obs containing sub-cluster information.
    new_label_col : str Name of the column to store the AI-generated subtype labels.

    Returns:
    --------
    tuple A tuple containing:
    AnnData: Concatenated annotated data with AI-generated subtype labels.
    dict: Mapping of original labels to AI-generated labels.
    """
    #build adata_dict based on cell_type_col
    adata_dict = build_adata_dict(adata, strata_keys=cell_type_col)

    label_mappings = ai_annotate_cell_type_by_comparison_adata_dict(adata_dict, groupby=sub_cluster_col, n_top_genes=n_top_genes, label_column=new_label_col, tissue_of_origin_col=tissue_of_origin_col, subtype=True)

    adata = concatenate_adata_dict(adata_dict, index_unique=None) #setting index_unique=None avoids index modification

    return adata, label_mappings