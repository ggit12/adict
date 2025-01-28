"""
Main init for anndictionary.
If on Mac, runs a multithreading configuration check before allowing import.
"""
import os
import platform
import sys
import numba  # Import numba to interact with its threading layer

#import AnnDictionary namespace
from . import utils
from . import adata_dict
from . import llm
from . import wrappers
from . import automated_label_management

from .adata_dict import (
    #adata_dict.py
    AdataDict,
    to_nested_tuple,
    to_nested_list,

    #adata_dict_fapply.py
    adata_dict_fapply,
    adata_dict_fapply_return,

    #read_write_build.py
    read_adata_dict,
    write_adata_dict,
    read_adata_dict_from_h5ad,
    read,
    build_adata_dict,
    add_stratification,
    concatenate_adata_dict,
    check_and_create_stratifier,

)

from .llm import (
    #llm_call.py
    configure_llm_backend,
    get_llm_config,
    call_llm,
    retry_call_llm,

    #parse_llm_response.py
    extract_dictionary_from_ai_string,
    extract_list_from_ai_string,
    process_llm_category_mapping,

    #llm_providers.py
    LLMProviders,

)

from .annotate import (

    # Genes
    #--------
    ai_gene_list,

    # Cells
    #--------
    # De Novo
    ai_annotate, ai_annotate_by_comparison,
    ai_cell_type, ai_annotate_cell_type,
    ai_cell_types_by_comparison,
    ai_annotate_cell_sub_type,
    ai_annotate_biological_process,
    cell_type_marker_gene_score,
    ai_determine_leiden_resolution,

    # Label Transfer
    harmony_label_transfer,
    stable_label, stable_label_adata,

    # Benchmarking
    create_label_df,
    kappa_adata, krippendorff_alpha_adata,
    ai_compare_cell_type_labels_pairwise,

)

from .plot import (
    #cells.py
    module_score_barplot,
    module_score_umap,

    #genes.py
    annotate_gene_groups_with_ai_biological_process,

    #benchmarking.py
    plot_training_history,
    plot_changes,
    plot_confusion_matrix_from_adata,
    plot_confusion_matrix,
    plot_sankey,
    save_sankey,
    plot_grouped_average,
    plot_model_agreement,
    plot_model_agreement_categorical,

)

from .automated_label_management import (

    #clean_single_column.py
    simplify_obs_column,
    simplify_var_index,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,
    map_gene_labels_to_simplified_set,

    #unify_columns_within_adata.py
    ensure_label_consistency_main,
    ensure_label_consistency_adata,

    #unify_columns_between_adata.py
    ai_unify_labels,

)


__all__ = [
    # Core
    # -----------
    #adata_dict.py
    "AdataDict",
    "to_nested_tuple",
    "to_nested_list",

    #adata_dict_fapply.py
    "adata_dict_fapply",
    "adata_dict_fapply_return",

    #read_write_build.py
    "read_adata_dict",
    "write_adata_dict",
    "read_adata_dict_from_h5ad",
    "read",
    "build_adata_dict",
    "add_stratification",
    "concatenate_adata_dict",
    "check_and_create_stratifier",


    # LLM
    # --------
    #llm_call.py
    "configure_llm_backend",
    "get_llm_config",
    "call_llm",
    "retry_call_llm",

    #parse_llm_response.py
    "extract_dictionary_from_ai_string",
    "extract_list_from_ai_string",
    "process_llm_category_mapping",

    #llm_providers.py
    "LLMProviders",


    # Annotate
    # -----------
    # De Novo
    "ai_annotate",
    "ai_annotate_by_comparison",
    "ai_cell_type",
    "ai_annotate_cell_type",
    "ai_cell_types_by_comparison",
    "ai_annotate_cell_sub_type",
    "ai_annotate_biological_process",
    "cell_type_marker_gene_score",
    "ai_determine_leiden_resolution",

    # Label Transfer
    "harmony_label_transfer",
    "stable_label",
    "stable_label_adata",

    # Benchmarking
    "create_label_df",
    "kappa_adata",
    "krippendorff_alpha_adata",
    "ai_compare_cell_type_labels_pairwise",


    # Plot
    # -------
    # from cells.py
    "module_score_barplot",
    "module_score_umap",

    # from genes.py
    "annotate_gene_groups_with_ai_biological_process",

    # from benchmarking.py
    "plot_training_history",
    "plot_changes",
    "plot_confusion_matrix_from_adata",
    "plot_confusion_matrix",
    "plot_sankey",
    "save_sankey",
    "plot_grouped_average",
    "plot_model_agreement",
    "plot_model_agreement_categorical",

    # Automated Label Management
    # ----------------------------
    # from clean_single_column.py
    "simplify_obs_column",
    "simplify_var_index",
    "create_label_hierarchy",
    "map_cell_type_labels_to_simplified_set",
    "map_gene_labels_to_simplified_set",

    # from unify_columns_within_adata.py
    "ensure_label_consistency_main",
    "ensure_label_consistency_adata",

    # from unify_columns_between_adata.py
    "ai_unify_labels",

]


# Run mac system check for multithreading compatibility
if platform.system() == "Darwin":
    try:
        numba.config.THREADING_LAYER = 'tbb'
        # numba.set_num_threads(2)

        @numba.jit(nopython=True, parallel=True)
        def _test_func():
            acc = 0
            for i in numba.prange(4): # pylint: disable=not-an-iterable
                acc += i
            return acc

        _test_func()
        if numba.config.THREADING_LAYER != 'tbb':
            raise RuntimeError("Expected TBB threading layer, got something else.")

    except Exception:
        # Print only our custom error and exit; no traceback will be shown.
        sys.tracebacklimit = 0  # Suppress traceback
        raise RuntimeError(
            "Failed to initialize TBB threading layer on macOS!\n"
            "Try re-installing numba + TBB via conda (run exactly these 3 lines of code):\n"
            "  pip uninstall tbb numba\n"
            "  conda remove tbb numba\n"
            "  conda install -c conda-forge tbb numba\n"
            "Then restart python and re-attempt import\n"
        ) from None


# Run mac system configuration for multithreading
if platform.system() == 'Darwin':
    # Set Numba threading layer to 'tbb'
    if os.getenv("NUMBA_THREADING_LAYER") is None:
        os.environ["NUMBA_THREADING_LAYER"] = "tbb"
        numba.config.THREADING_LAYER = 'tbb'  # Explicitly set the threading layer using config
