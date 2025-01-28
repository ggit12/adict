"""
This module contains the functions write AdataDict objects to disk.
"""

import os
import json
import scanpy as sc

from .adata_dict import AdataDict

def write_adata_dict(
    adata_dict: AdataDict,
    directory: str,
    *,
    file_prefix: str = "",
) -> None:
    """
    Save each :class:`AnnData` object from an :class:`AdataDict` into a separate ``.h5ad`` file, 
    creating a directory structure that reflects the hierarchy of the :class:`AdataDict`
    using key values as directory names. The hierarchy is saved in a file called 
    ``adata_dict.hierarchy`` in the top-level directory.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`.

    directory
        Base directory where ``.h5ad`` files will be saved.

    file_prefix
        Optional prefix for the filenames.

    Notes
    -----
    The directory structure uses key values as directory names, and the full key tuple 
    as the filename of the ``.h5ad`` file.

    Example
    -------
    If the hierarchy is ``('Donor', 'Tissue')`` and the adata_dict is:

    .. code-block:: python

        {
            ("Donor1", "Tissue1"): adata_d1_t1,
            ("Donor1", "Tissue2"): adata_d1_t2,
            ("Donor2", "Tissue1"): adata_d2_t1,
        }

    The files will be saved with the following directory structure:

    .. code-block:: text

        directory/
        adata_dict.hierarchy
            Donor1/
                Tissue1/
                    Donor1_Tissue1.h5ad
                Tissue2/
                    Donor1_Tissue2.h5ad
            Donor2/
                Tissue1/
                    Donor2_Tissue1.h5ad
    """

    # Create the base directory, throwing error if it exists already (to avoid overwriting)
    os.makedirs(directory, exist_ok=False)

    # Save the hierarchy to a file in the top-level directory
    hierarchy_file_path = os.path.join(directory, "adata_dict.hierarchy")
    with open(hierarchy_file_path, "w", encoding="utf-8") as f:
        # Save the hierarchy using JSON for easy reconstruction
        json.dump(adata_dict.hierarchy, f)

    # Flatten the AdataDict to get all AnnData objects with their keys
    flat_dict = adata_dict.flatten()

    # Iterate over the flattened dictionary and save each AnnData object
    for key, adata in flat_dict.items():
        # Build the path according to the key values (without hierarchy names)
        path_parts = [directory] + [str(k) for k in key]
        # Create the directory path
        dir_path = os.path.join(*path_parts)
        os.makedirs(dir_path, exist_ok=True)
        # Construct the filename using the full key tuple
        filename = f"{file_prefix}{'_'.join(map(str, key))}.h5ad"
        file_path = os.path.join(dir_path, filename)
        # Save the AnnData object
        sc.write(file_path, adata)
