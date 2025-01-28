"""
This module contains the functions necessary to build AdataDict objects from adata in memory.
"""

import itertools

import pandas as pd
from anndata import AnnData

from .adata_dict import AdataDict, to_nested_list
from .adata_dict_fapply import adata_dict_fapply_return


def build_adata_dict(
    adata: AnnData,
    strata_keys: list[str],
    *,
    desired_strata: list | dict | None = None,
) -> AdataDict:
    """
    Build a dictionary of :class:`AnnData` objects split by the ``obs`` columns specified in ``strata_keys``.

    Parameters
    ------------
    adata
        The input :class:`AnnData`.

    strata_keys
        List of column names in ``adata.obs`` to use for stratification.

    desired_strata
        List of desired strata tuples or a dictionary where keys are the col names supplied in ``strata_keys``, and values 
        are lists of desired strata. If None (Default), all combinations of categories 
        in ``adata.obs[strata_keys]`` will be used.

    Returns
    --------
    Flat dictionary of class :class:`AdataDict` where keys are strata tuples and values are corresponding :class:`AnnData` subsets.

    Raises
    ------
    ValueError
        If `desired_strata` is neither a list nor a dictionary of lists.

    Examples
    --------
        Case 1: Take all groups
            >>> import pandas as pd
            >>> from anndata import AnnData
            >>> # Create an example AnnData object
            >>> adata = AnnData(obs=pd.DataFrame({
            ...     "Donor": ["Donor1", "Donor1", "Donor2"],
            ...     "Tissue": ["Tissue1", "Tissue2", "Tissue1"]
            ... }))
            >>> strata_keys = ["Donor", "Tissue"]
            >>> adata_dict = build_adata_dict(adata, strata_keys)
            >>> print(adata_dict)
            {
                ("Donor1", "Tissue1"): adata_d1_t1,
                ("Donor1", "Tissue2"): adata_d1_t2,
                ("Donor2", "Tissue1"): adata_d2_t1,
            }
        Case 2: Take only some groups
            >>> import pandas as pd
            >>> from anndata import AnnData
            >>> # Create an example AnnData object
            >>> adata = AnnData(obs=pd.DataFrame({
            ...     "Donor": ["Donor1", "Donor1", "Donor2"],
            ...     "Tissue": ["Tissue1", "Tissue2", "Tissue1"]
            ... }))
            >>> strata_keys = ["Donor", "Tissue"]
            >>> desired_strata = {
                                    "Donor": ["Donor1"], #Take only Donor1
                                    "Tissue": ["Tissue1", "Tissue2"] #Take Tissue1 and Tissue2
            }
            >>> adata_dict = build_adata_dict(adata, strata_keys, desired_strata=desired_strata)
            >>> print(adata_dict)
            {
                ("Donor1", "Tissue1"): adata_d1_t1,
                ("Donor1", "Tissue2"): adata_d1_t2
            }
    """
    if desired_strata is None:
        # Generate all combinations of categories in adata.obs[strata_keys]
        all_categories = [adata.obs[key].cat.categories.tolist() for key in strata_keys]
        all_combinations = list(itertools.product(*all_categories))
        desired_strata = all_combinations
        return build_adata_dict_main(
            adata, strata_keys, desired_strata, print_missing_strata=False
        )

    elif isinstance(desired_strata, list):
        # Ensure that desired_strata is a list of tuples
        if all(isinstance(item, str) for item in desired_strata):
            raise ValueError("desired_strata should be a list of tuples, not strings.")
        return build_adata_dict_main(adata, strata_keys, desired_strata)

    elif isinstance(desired_strata, dict):
        # Generate all combinations of desired strata values across strata_keys
        all_combinations = itertools.product(
            *(desired_strata[key] for key in strata_keys)
        )
        desired_strata = list(all_combinations)
        return build_adata_dict_main(adata, strata_keys, desired_strata)

    else:
        raise ValueError(
            "desired_strata must be either a list of tuples or a dictionary of lists"
        )


def build_adata_dict_main(
    adata: AnnData,
    strata_keys: list[str],
    desired_strata: list[tuple],
    *,
    print_missing_strata: bool = True,
) -> AdataDict:
    """
    Function to build a dictionary of AnnData objects based on desired strata values.

    Parameters
    ------------
    adata
        Annotated data matrix.

    strata_keys
        List of column names in `adata.obs` to use for stratification.

    desired_strata
        List of desired strata tuples.

    print_missing_strata
        If True, print missing strata. Default is True.

    Returns
    -------
    Flat dictionary of class :class:`AdataDict` where keys are strata tuples and values are corresponding AnnData subsets.
    """
    # Ensure that the strata columns are categorical
    for key in strata_keys:
        if not pd.api.types.is_categorical_dtype(adata.obs[key]):
            adata.obs[key] = adata.obs[key].astype("category")

    # Group indices by combinations of strata_keys for efficient access
    groups = adata.obs.groupby(strata_keys, observed=False).indices

    # Adjust group keys to always be tuples
    if len(strata_keys) == 1:
        groups = {(k,): v for k, v in groups.items()}

    # Initialize the dictionary to store subsets
    adata_dict = {}

    # Iterate over desired strata (tuples) and extract subsets
    for stratum in desired_strata:
        if stratum in groups:
            indices = groups[stratum]
            adata_dict[stratum] = adata[indices].copy()
        else:
            if print_missing_strata:
                print(
                    f"Warning: No {stratum} in data based on {strata_keys}."
                )

    # Create AdataDict and set hierarchy to strata_keys
    adata_dict = AdataDict(adata_dict, tuple(strata_keys))
    return adata_dict


def add_stratification(
    adata_dict: AdataDict,
    strata_keys: list[str],
    *,
    desired_strata: list | dict | None = None,
) -> AdataDict:
    """
    Split each value of an AnnData dictionary into further subsets based on additional desired strata.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`

    strata_keys
        List of column names in `adata.obs` to use for further stratification.

    desired_strata
        List of desired strata values or a dictionary where keys are strata keys and values 
        are lists of desired strata values.

    Returns
    -------
    Nested :class:`AdataDict`, where the top-level is now stratified by ``strata_keys``.

    Raises
    ------
    ValueError
        If any of the `strata_keys` are already in the hierarchy.

    Examples
    --------
        Case 1: Build by Donor first, then add Tissue stratification after
            >>> import pandas as pd
            >>> from anndata import AnnData
            >>> # Create an example AnnData object
            >>> adata = AnnData(obs=pd.DataFrame({
            ...     "Donor": ["Donor1", "Donor1", "Donor2"],
            ...     "Tissue": ["Tissue1", "Tissue2", "Tissue1"]
            ... }))
            >>> # First, build an AdataDict grouped/stratified by Donor
            >>> strata_keys = ["Donor"]
            >>> adata_dict = build_adata_dict(adata, strata_keys)
            >>> print(adata_dict)
            {
                ("Donor1",): adata_d1,
                ("Donor2",): adata_d2,
            }
            >>> # Then, add a stratification by Tissue
            >>> strata_keys = ["Tissue"]
            >>> add_stratification(adata_dict, strata_keys)
            >>> print(adata_dict)
            {
                ("Tissue1",) : {
                    ("Donor1",) : adata_d1_t1,
                    ("Donor2",) : adata_d2_t1
                    },
                ("Tissue2",) : {
                    ("Donor1",) : adata_d1_t2
                    }
            }
            >>> # Note, we can always flatten or rearrange the nesting structure
            >>> adata_dict.set_hierarchy(["Donor","Tissue"])
            >>> print(adata_dict)
            {
                ("Donor1", "Tissue1"): adata_d1_t1,
                ("Donor1", "Tissue2"): adata_d1_t2,
                ("Donor2", "Tissue1"): adata_d2_t1,
            }
            >>> #For example, if we want Donor as the top-level index
            >>> adata_dict.set_hierarchy(["Donor",["Tissue"]])
                        {
                ("Donor1",) : {
                    ("Tissue1",) : adata_d1_t1,
                    ("Tissue2",) : adata_d1_t2
                    },
                ("Donor2",) : {
                    ("Tissue1",) : adata_d2_t1
                    }
            }
    """
    # Get the hierarchy and check for redundant stratification
    cached_hierarchy = adata_dict.hierarchy

    for strata_key in strata_keys:
        if strata_key in adata_dict.flatten_nesting_list(to_nested_list(cached_hierarchy)):
            raise ValueError(f"adata_dict is already stratified by {strata_key}")

    # Flatten the adata_dict using the class method
    adata_dict = adata_dict.flatten()

    # Split the adata_dict
    adata_dict = adata_dict_fapply_return(adata_dict, build_adata_dict, 
                                         strata_keys=strata_keys, 
                                         desired_strata=desired_strata)

    # Create new hierarchy with strata_keys at top level
    new_hierarchy = [strata_keys, to_nested_list(cached_hierarchy)]

    # Set the new hierarchy
    adata_dict.set_hierarchy(new_hierarchy)

    return adata_dict
