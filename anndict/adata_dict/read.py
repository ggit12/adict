"""
This module contains the functions necessary to read :class:`AdataDict` objects from adata on disk.
"""

import os
import json
from collections import Counter

import anndata as ad
import scanpy as sc

from .adata_dict import AdataDict, to_nested_tuple

def read_adata_dict(
    directory: str,
) -> AdataDict:
    """
    Read an :class:`AdataDict` from a previously saved :class:`AdataDict`. To write an :class:`AdataDict` see :func:`~write_adata_dict`.

    Parameters
    -----------
    directory
        Base directory where the ``.h5ad`` files and hierarchy file are located.

    Returns
    --------
    An :class:`AdataDict` reconstructed from the saved files.

    Example
    --------
    If the directory structure is:

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

    And the hierarchy was:

    .. code-block:: text

        ("Donor", "Tissue")

    The reconstructed adata_dict will be:

    .. code-block:: python

        {
            ("Donor1", "Tissue1"): adata_d1_t1,
            ("Donor1", "Tissue2"): adata_d1_t2,
            ("Donor2", "Tissue1"): adata_d2_t1,
        }
    """

    # Read the hierarchy from the file
    hierarchy_file_path = os.path.join(directory, "adata_dict.hierarchy")
    with open(hierarchy_file_path, "r", encoding="utf-8") as f:
        # tuples will be converted to lists on write, so need to convert back to tuple on load
        hierarchy = to_nested_tuple(json.load(f))

    # Initialize an empty AdataDict with the hierarchy
    adata_dict = AdataDict(hierarchy=hierarchy)

    # Function to recursively rebuild the nested AdataDict
    def add_to_adata_dict(current_dict, key_tuple, adata):
        """
        Recursively adds the AnnData object to the appropriate place in the nested AdataDict.

        Parameters
        ------------
        current_dict
            The current level of AdataDict.
        key_tuple
            Tuple of key elements indicating the path.
        adata
            The AnnData object to add.
        """
        if len(key_tuple) == 1:
            current_dict[key_tuple[0]] = adata
        else:
            key = key_tuple[0]
            if key not in current_dict:
                current_dict[key] = AdataDict(hierarchy=hierarchy[1:])
            add_to_adata_dict(current_dict[key], key_tuple[1:], adata)

    # Walk through the directory structure
    for root, dirs, files in os.walk(directory):
        # Skip the top-level directory where the hierarchy file is located
        relative_path = os.path.relpath(root, directory)
        if relative_path == ".":
            continue
        for file in files:
            if file.endswith(".h5ad"):
                # Reconstruct the key from the directory path
                path_parts = relative_path.split(os.sep)
                key_elements = path_parts
                # Remove empty strings (if any)
                key_elements = [k for k in key_elements if k]
                key = tuple(key_elements)
                # Read the AnnData object
                file_path = os.path.join(root, file)
                adata = sc.read(file_path)
                # Add to the AdataDict
                add_to_adata_dict(adata_dict, key, adata)
    return adata_dict


def read(
    directory_list: str | list[str],
    *,
    keys: list[str] | None = None,
) -> AdataDict:
    """
    Process a list of directories or file paths to read AnnData objects. 

    For each directory, if a `.hierarchy` file is found, the directory is processed 
    with `read_adata_dict`. Otherwise, the highest-level directory is processed with 
    `read_adata_dict_from_h5ad`. Subdirectories of a directory containing a `.hierarchy` 
    file are not processed with `read_adata_dict_from_h5ad`.

    Parameters
    ------------
    directory_list
        String or list of strings, paths to directories or `.h5ad` files.

    keys
        List of strings that will be used as keys for the resulting dictionary.

    Returns
    -------
    An :class:`AdataDict` of all AnnData objects.
    """
    if isinstance(directory_list, str):
        directory_list = [directory_list]

    adata_dict = {}

    # Set to keep track of directories that have been processed with read_adata_dict
    hierarchy_dirs = set()

    # List to collect .h5ad files to process
    h5ad_files = []

    # Function to find all directories containing adata_dict.hierarchy files
    def find_hierarchy_dirs(dir_path):
        for root, dirs, files in os.walk(dir_path):
            if "adata_dict.hierarchy" in files:
                hierarchy_dirs.add(root)
                # Do not traverse subdirectories of directories with hierarchy files
                dirs[:] = []
            else:
                # Continue traversing subdirectories
                pass

    # First, process the input paths to find hierarchy directories and collect .h5ad files
    for path in directory_list:
        if os.path.isfile(path):
            if path.endswith(".h5ad"):
                h5ad_files.append(path)
        elif os.path.isdir(path):
            # Find hierarchy directories
            find_hierarchy_dirs(path)
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory.")

    # Process directories with hierarchy files using read_adata_dict
    for h_dir in hierarchy_dirs:
        adata_dict.update(read_adata_dict(h_dir))

    # Build a set of directories to exclude (hierarchy_dirs and their subdirectories)
    exclude_dirs = set()
    for h_dir in hierarchy_dirs:
        for root, dirs, files in os.walk(h_dir):
            exclude_dirs.add(root)

    # Function to collect .h5ad files not under exclude_dirs
    def collect_h5ad_files(dir_path):
        for root, dirs, files in os.walk(dir_path):
            # Skip directories under exclude_dirs
            if any(
                os.path.commonpath([root, excl_dir]) == excl_dir
                for excl_dir in exclude_dirs
            ):
                dirs[:] = []
                continue
            for file in files:
                if file.endswith(".h5ad"):
                    h5ad_files.append(os.path.join(root, file))

    # Collect .h5ad files from directories not containing hierarchy files
    for path in directory_list:
        if os.path.isdir(path):
            collect_h5ad_files(path)

    # Process the collected .h5ad files using read_adata_dict_from_h5ad
    if h5ad_files:
        adata_dict.update(read_adata_dict_from_h5ad(h5ad_files, keys=keys))

    return adata_dict


def read_adata_dict_from_h5ad(
    paths: str | list[str],
    *,
    keys: list[str] | None = None,
) -> AdataDict:
    """
    Read ``.h5ad`` files from a list of paths and return them in a dictionary.

    For each element in the provided list of paths, if the element is a directory, 
    it reads all ``.h5ad`` files in that directory. If the element is an ``.h5ad`` file, 
    it reads the file directly.

    For auto-generated keys, if there are duplicate filenames, the function will 
    include parent directory names from right to left until keys are unique. 
    For example, ``dat/heart/fibroblast.h5ad`` would generate the key ``('heart', 'fibroblast')``
    if disambiguation is needed.

    Parameters
    ------------
    paths
        A string path or list of paths to directories or ``.h5ad`` files.

    keys
        A list of strings to use as keys for the adata_dict. If provided, must be equal 
        in length to the number of ``.h5ad`` files read.

    Returns
    -------
    An :class:`AdataDict`.
    """

    if isinstance(paths, str):
        paths = [paths]

    adata_dict = {}
    file_paths = []

    # First, collect all file paths
    for path in paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".h5ad"):
                    file_paths.append(os.path.join(path, file))
        elif path.endswith(".h5ad"):
            file_paths.append(path)

    # Check if provided keys match the number of files
    if keys is not None:
        if len(keys) != len(file_paths):
            raise ValueError(
                f"Number of provided keys ({len(keys)}) does not match the number of .h5ad files ({len(file_paths)})"
            )
        # Check for uniqueness in provided keys
        key_counts = Counter(keys)
        duplicates = [k for k, v in key_counts.items() if v > 1]
        if duplicates:
            raise ValueError(f"Duplicate keys found: {duplicates}")
        # Convert provided keys to tuples
        tuple_keys = [tuple(k) if isinstance(k, (list, tuple)) else (k,) for k in keys]
    else:
        # Generate keys from paths
        base_names = [os.path.splitext(os.path.basename(fp))[0] for fp in file_paths]

        # Start with just the base names
        tuple_keys = [(name,) for name in base_names]

        # Keep extending paths to the left until all keys are unique
        while len(set(tuple_keys)) != len(tuple_keys):
            new_tuple_keys = []
            for i, file_path in enumerate(file_paths):
                path_parts = os.path.normpath(file_path).split(os.sep)
                # Find the current key's elements in the path
                current_key = tuple_keys[i]
                current_idx = (
                    len(path_parts) - 1 - len(current_key)
                )  # -1 for zero-based index
                # Add one more path element to the left if possible
                if current_idx > 0:
                    new_key = (path_parts[current_idx - 1],) + current_key
                else:
                    new_key = current_key
                new_tuple_keys.append(new_key)
            tuple_keys = new_tuple_keys

            # Safety check - if we've used all path components and still have duplicates
            if all(
                len(key) == len(os.path.normpath(fp).split(os.sep))
                for key, fp in zip(tuple_keys, file_paths)
            ):
                raise ValueError("Unable to create unique keys even using full paths")

    # Process the files with the finalized tuple keys
    for i, file_path in enumerate(file_paths):
        adata_dict[tuple_keys[i]] = ad.read_h5ad(file_path)

    return adata_dict
