"""
This module contains the adata_dict_fapply family of functions, the core functions of Anndictionary.
"""

import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from .adata_dict import AdataDict


def apply_func(adt_key, adata, func, accepts_key, max_retries, **func_args):
    """
    Applies a function to data with retries (max_retries many times), optionally passing a key.
    """
    attempts = -1
    while attempts < max_retries:
        try:
            if accepts_key:
                func(adata, adt_key=adt_key, **func_args)
            else:
                func(adata, **func_args)
            return  # Success, exit the function
        except Exception as e:
            attempts += 1
            print(f"Error processing {adt_key} on attempt {attempts}: {e}")
            if attempts >= max_retries:
                print(f"Failed to process {adt_key} after {max_retries} attempts.")


def adata_dict_fapply(
    adata_dict: AdataDict,
    func: callable,
    *,
    use_multithreading: bool = True,
    num_workers: int | None = None,
    max_retries: int = 0,
    **kwargs_dicts: dict,
) -> None:
    """
    Applies a given function to each AnnData object in the ``adata_dict``, with error handling,
    a retry mechanism, and the option to use either multithreading or sequential execution.

    Parameters
    ------------
    adata_dict
        An object of class :class:`AdataDict`.  

    func
        Function to apply to each AnnData object in the dictionary.

    use_multithreading
        If True, use ``ThreadPoolExecutor``; if False, execute sequentially. Default is True.

    num_workers
        Number of worker threads to use. If :obj:`None`, defaults to the number of CPUs available.

    max_retries
        Maximum number of retries for a failed task. Default is 3.

    kwargs_dicts
        Additional keyword arguments to pass to the function.

    Returns
    -------
    None
        ``adata_dict_fapply`` applies ``func`` to each item in ``adata_dict``,
        potentially modifying the AnnData objects in place.
    """

    sig = inspect.signature(func)
    accepts_key = "adt_key" in sig.parameters

    def get_arg_value(arg_value, adt_key):
        if isinstance(arg_value, dict):
            if adt_key in arg_value:
                return arg_value[adt_key]
            elif not set(adata_dict.keys()).issubset(arg_value.keys()):
                # Use the entire dictionary if it doesn't contain all adata_dict keys
                return arg_value
        # Use the value as is if it's not a dictionary or doesn't contain all adata_dict keys
        return arg_value

    if use_multithreading:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    apply_func,
                    adt_key,
                    adata,
                    func,
                    accepts_key,
                    max_retries,
                    **{
                        arg_name: get_arg_value(arg_value, adt_key)
                        for arg_name, arg_value in kwargs_dicts.items()
                    },
                ): adt_key
                for adt_key, adata in adata_dict.items()
            }

            for future in as_completed(futures):
                adt_key = futures[future]
                try:
                    future.result()  # Retrieve result to catch exceptions
                except Exception as e:
                    print(f"Unhandled error processing {adt_key}: {e}")
    else:
        for adt_key, adata in adata_dict.items():
            try:
                apply_func(
                    adt_key,
                    adata,
                    func,
                    accepts_key,
                    max_retries,
                    **{
                        arg_name: get_arg_value(arg_value, adt_key)
                        for arg_name, arg_value in kwargs_dicts.items()
                    },
                )
            except Exception as e:
                print(f"Unhandled error processing {adt_key}: {e}")


def apply_func_return(adt_key, adata, func, accepts_key, max_retries, **func_args):
    """
    Applies a function to data with retries (max_retries many times), optionally passing a key.
    Returns the return of func.
    """
    attempts = -1
    while attempts < max_retries:
        try:
            if accepts_key:
                return func(adata, adt_key=adt_key, **func_args)
            else:
                return func(adata, **func_args)
        except Exception as e:
            attempts += 1
            print(f"Error processing {adt_key} on attempt {attempts}: {e}")
            if attempts >= max_retries:
                print(f"Failed to process {adt_key} after {max_retries} attempts.")
                return f"Error: {e}"  # Optionally, return None or raise an error


def adata_dict_fapply_return(
    adata_dict: AdataDict,
    func: callable,
    *,
    use_multithreading: bool = True,
    num_workers: int | None = None,
    max_retries: int = 0,
    return_as_adata_dict: bool = False,
    **kwargs_dicts: dict,
) -> dict | AdataDict:
    """
    Applies a given function to each AnnData object in the adata_dict, with error handling,
    retry mechanism, and the option to use either threading or sequential execution. Returns
    a dictionary with the results of the function applied to each AnnData object.

    Parameters
    -------------
    adata_dict
        Dictionary of AnnData objects with keys as identifiers.
    func
        Function to apply to each AnnData object in the dictionary.
    use_multithreading
        If True, use ThreadPoolExecutor; if False, execute sequentially.
    num_workers
        Number of worker threads to use (default: number of CPUs available).
    max_retries
        Maximum number of retries for a failed task.
    kwargs_dicts
        Additional keyword arguments to pass to the function.

    Returns
    ----------
    A :class:`dict` with the same keys as ``adata_dict``, containing the results of ``func`` applied to each AnnData in ``adata_dict``. Can optionally be coerced to :class:`AdataDict` by setting ``return_as_adata_dict=True``.
    """
    sig = inspect.signature(func)
    accepts_key = "adt_key" in sig.parameters
    results = {}

    if return_as_adata_dict:
        # if not isinstance(adata_dict, AdataDict):
        #     raise ValueError("You cannot return as class AdataDict if input is not already of class AdataDict")
        hierarchy = adata_dict.hierarchy if hasattr(adata_dict, "hierarchy") else ()

    def get_arg_value(arg_value, adt_key):
        if isinstance(arg_value, dict):
            if adt_key in arg_value:
                return arg_value[adt_key]
            elif not set(adata_dict.keys()).issubset(arg_value.keys()):
                # Use the entire dictionary if it doesn't contain all adata_dict keys
                return arg_value
        # Use the value as is if it's not a dictionary or doesn't contain all adata_dict keys
        return arg_value

    if use_multithreading:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    apply_func_return,
                    adt_key,
                    adata,
                    func,
                    accepts_key,
                    max_retries,
                    **{
                        arg_name: get_arg_value(arg_value, adt_key)
                        for arg_name, arg_value in kwargs_dicts.items()
                    },
                ): adt_key
                for adt_key, adata in adata_dict.items()
            }

            for future in as_completed(futures):
                adt_key = futures[future]
                try:
                    result = future.result()  # Retrieve result to catch exceptions
                    results[adt_key] = result
                except Exception as e:
                    print(f"Unhandled error processing {adt_key}: {e}")
                    results[adt_key] = (
                        None  # Optionally, return None or handle differently
                    )
    else:
        for adt_key, adata in adata_dict.items():
            try:
                result = apply_func_return(
                    adt_key,
                    adata,
                    func,
                    accepts_key,
                    max_retries,
                    **{
                        arg_name: get_arg_value(arg_value, adt_key)
                        for arg_name, arg_value in kwargs_dicts.items()
                    },
                )
                results[adt_key] = result
            except Exception as e:
                print(f"Unhandled error processing {adt_key}: {e}")
                results[adt_key] = None  # Optionally, return None or handle differently

    if return_as_adata_dict:
        results = AdataDict(results, hierarchy)

    return results
