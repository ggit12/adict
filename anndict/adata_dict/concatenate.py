"""
Concatenate :class:`AdataDict` back to a single :class:`AnnData`
"""

import anndata as ad

from anndata import AnnData

from .adata_dict import AdataDict
from .adata_dict_fapply import adata_dict_fapply


def concatenate_adata_dict(
    adata_dict: AdataDict,
    *,
    new_col_name: str | None = None,
    **kwargs,
) -> AnnData:
    """
    Concatenate all AnnData objects in `adata_dict` into a single AnnData object. 
    If only a single AnnData object is present, return it as is.

    Parameters
    ------------
    adata_dict
        :class:`AdataDict`

    new_col_name
        If provided, the name of the new column that will store the ``adata_dict`` 
        key in ``.obs`` of the concatenated AnnData. Defaults to None.

    kwargs
        Additional keyword arguments for concatenation.

    Returns
    -------
    A single :class:`AnnData` object. 
    The ``.obs`` will contain a new column specifying the key of the :class:`AnnData` of origin.
    """
    kwargs.setdefault("join", "outer")
    kwargs.setdefault("index_unique", None)  # Ensure original indices are kept

    adatas = list(adata_dict.values())

    # add the key to the obs to keep track after merging
    def add_key_to_obs_adata_dict(adata_dict, new_col_name=new_col_name):
        def add_key_to_obs_adata(adata, new_col_name=new_col_name, adt_key=None):
            adata.obs[new_col_name] = [adt_key] * adata.n_obs

        adata_dict_fapply(adata_dict, add_key_to_obs_adata)

    if new_col_name:
        add_key_to_obs_adata_dict(adata_dict)

    if len(adatas) == 1:
        return adatas[0]  # Return the single AnnData object as is

    if adatas:
        return ad.concat(adatas, **kwargs)
    else:
        raise ValueError("adata_dict is empty. No data available to concatenate.")
