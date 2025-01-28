#squidpy adata_dict functions
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import re
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import squidpy as sq

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import dok_matrix

from .dict import adata_dict_fapply


def read_data(file_path, platform=None):
    """
    Reads the data from a CSV file and ensures it contains the necessary columns.

    Parameters:
    file_path (str): The path to the CSV file.
    platform (str): The platform type ("Merscope" or "Xenium").

    Returns:
    pd.DataFrame: The data read from the file.

    Raises:
    ValueError: If the required columns are not present in the file or if the file format is unsupported.
    """
    print("reading data")

    if not file_path.endswith('.csv'):
        raise ValueError("Unsupported file format. Please provide a CSV file.")

    df = pd.read_csv(file_path)

    required_columns_merscope = {'global_x', 'global_y', 'gene'}
    required_columns_xenium = {'feature_name', 'x_location', 'y_location'}

    if platform == "Merscope":
        if not required_columns_merscope.issubset(df.columns):
            raise ValueError(f"The file must contain the following columns for Merscope: {required_columns_merscope}")
    elif platform == "Xenium":
        if not required_columns_xenium.issubset(df.columns):
            raise ValueError(f"The file must contain the following columns for Xenium: {required_columns_xenium}")
        df = df.rename(columns={"feature_name": "gene", "x_location": "global_x", "y_location": "global_y"})
    else:
        raise ValueError("Unsupported platform. Please provide either 'Merscope' or 'Xenium' as the platform.")

    return df



def get_steps_and_coords(df, box_size, step_size):
    """
    Computes the number of steps and the top-left coordinates of each box.

    Parameters:
    df (pd.DataFrame): The data containing 'global_x' and 'global_y' columns.
    box_size (int): The size of the box.
    step_size (int): The step size.

    Returns:
    tuple: A tuple containing the number of steps in x and y directions, and the list of top-left coordinates of each box.
    
    Raises:
    ValueError: If the box size is larger than the image dimensions.
    """
    print("getting steps and coords")
    min_x, max_x = df['global_x'].min(), df['global_x'].max()
    min_y, max_y = df['global_y'].min(), df['global_y'].max()
    
    x_steps = int((max_x - min_x - box_size) / step_size) + 2
    y_steps = int((max_y - min_y - box_size) / step_size) + 2
    
    if x_steps < 0:
        raise ValueError("box size is larger than image")
    
    coords_top_left = [[min_x + step_size*i, min_y + step_size*j] for i in range(0, x_steps) for j in range(0, y_steps)]
    
    return x_steps, y_steps, coords_top_left


def populate_sparse_array(df, coords_top_left, genes, step_size):
    """
    Populates a sparse array with gene counts.

    Parameters:
    df (pd.DataFrame): The data containing 'global_x', 'global_y', and 'gene' columns.
    coords_top_left (list): The list of top-left coordinates of each box.
    genes (np.array): The unique genes.
    step_size (int): The step size.

    Returns:
    scipy.sparse.csr_matrix: The sparse matrix with gene counts.
    """
    num_boxes_x = int((df['global_x'].max() - df['global_x'].min()) // step_size) + 1
    num_boxes_y = int((df['global_y'].max() - df['global_y'].min()) // step_size) + 1
    num_boxes = num_boxes_x * num_boxes_y

    sparse_array = dok_matrix((num_boxes, len(genes)), dtype=np.int32)
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    
    min_x, min_y = df['global_x'].min(), df['global_y'].min()
    df['box_x'] = ((df['global_x'] - min_x) // step_size).astype(int)
    df['box_y'] = ((df['global_y'] - min_y) // step_size).astype(int)
    
    for (box_x, box_y), box_df in df.groupby(['box_x', 'box_y']):
        index = box_x * num_boxes_y + box_y
        if 0 <= index < num_boxes:
            gene_counts = box_df['gene'].value_counts()
            for gene, count in gene_counts.items():
                if gene in gene_to_index:
                    sparse_array[index, gene_to_index[gene]] = count

    return sparse_array.tocsr()


def process_gene_counts(file_path, box_size, step_size, platform=None):
    """
    Processes the gene counts from the CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    box_size (int): The size of the box.
    step_size (int): The step size.

    Returns:
    tuple: A tuple containing the sparse matrix, unique genes, and list of top-left coordinates of each box.
    """
    df = read_data(file_path, platform=platform)
    print("processing gene counts")
    genes = df['gene'].unique()
    x_steps, y_steps, coords_top_left = get_steps_and_coords(df, box_size, step_size)
    sparse_array = populate_sparse_array(df, coords_top_left, genes, step_size)
    return sparse_array, genes, coords_top_left


def create_anndata(sparse_array, genes, coords_top_left):
    """
    Creates an AnnData object from the sparse matrix and coordinates.

    Parameters:
    sparse_array (scipy.sparse.csr_matrix): The sparse matrix with gene counts.
    genes (np.array): The unique genes.
    coords_top_left (list): The list of top-left coordinates of each box.

    Returns:
    anndata.AnnData: The AnnData object containing the gene counts and metadata.
    """
    print("creating anndata")
    adata = ad.AnnData(X=sparse_array.tocsr(), var={'gene_symbols': genes})
    adata.var.index = adata.var['gene_symbols']
    
    metadata_df = pd.DataFrame(coords_top_left, columns=['global_x_topleft', 'global_y_topleft'])
    adata.obs = metadata_df
    
    return adata

def add_blank_image_to_adata(adata, platform="Merscope"):
    """
    Adds a dummy image to the AnnData object based on the platform specifications.

    Parameters:
    adata (anndata.AnnData): The AnnData object to which the image will be added.
    platform (str): The platform from which the data originates. Default is "Merscope".

    Returns:
    anndata.AnnData: The updated AnnData object with the image and spatial data.
    """
    adata.obsm['spatial'] = adata.obs[['global_x_topleft', 'global_y_topleft']].to_numpy()
    
    max_x = int(adata.obs['global_x_topleft'].max())
    max_y = int(adata.obs['global_y_topleft'].max())
    
    dummy_image = np.ones((max_y + 1, max_x + 1, 3))
    
    if platform in ["Merscope", "Xenium"]:
        adata.uns['spatial'] = {
            'library_id': {
                'images': {
                    'hires': dummy_image,
                    'lowres': dummy_image
                },
                'scalefactors': {
                    'tissue_hires_scalef': 1.0,
                    'tissue_lowres_scalef': 1.0,
                    'spot_diameter_fullres': 16  # Example spot diameter, adjust as needed
                }
            }
        }
    
    return adata



def build_adata_from_transcript_positions(paths_dict, box_size=16, step_size=16, platform="Merscope"):
    """
    Builds an AnnData object from a detected_trancsripts.csv (Merscope) or transcripts.csv (Xenium) file and saves it to a specified output path. These are the files output by most spatial transcriptomic platforms, including Visium, Visium HD, Xenium, and Merscope.

    Parameters: 
    paths_dict (dict): A dictionary with input paths as keys and output paths as values.
    box_size (int, optional): The size of the box. Default is 16.
    step_size (int, optional): The step size. Default is 16.
    platform (str, optional): The platform used, either "Merscope" (default) or "Xenium".

    Returns:
    None

    Example:
    --------
    paths_dict = {
        'input_path1.csv': 'output_path1.h5ad',  

        'input_path2.csv': 'output_path2.h5ad'
        
        # Add more input-output path pairs as needed

        }

    build_adata_from_transcript_positions(paths_dict)

    """
    for input_path, output_path in paths_dict.items():
        sparse_array, genes, coords_top_left = process_gene_counts(input_path, box_size, step_size, platform=platform)
        adata = create_anndata(sparse_array, genes, coords_top_left)
        adata = add_blank_image_to_adata(adata, platform=platform)
        adata.write(output_path)

def build_adata_from_visium(paths_dict, hd=False):
    """
    Processes Visium data from input directories and saves the processed AnnData objects to specified output paths.

    Parameters:
    paths_dict (dict): A dictionary with input directories as keys and output file paths as values.
    hd (bool, optional): If True, converts 'spatial' to float and 'obs' columns to integers. Default is False.

    Returns:
    None

    Example:
    --------
    paths_dict = {
        'input_dir1': 'output_path1.h5ad',

        'input_dir2': 'output_path2.h5ad'
        
        # Add more input-output path pairs as needed

        }


    build_adata_from_visium(paths_dict, hd=True)
    """
    for input_dir, output_path in paths_dict.items():
        adata = sc.read_visium(input_dir)
        
        if hd:
            #Run fixes to sc.read_visium for visium HD data
            adata.obsm['spatial'] = adata.obsm['spatial'].astype(float)
            adata.obs = adata.obs.astype(int)
        
        adata.write(output_path)



def plot_spatial_adata_dict(adata_dict, **kwargs):
    """
    Plots spatial data for each AnnData object in adata_dict, colored by a specified variable.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'color_by' which specifies a variable by which to color the spatial plots, typically a column in .obs, and 'crop_coord' which specifies coordinates for cropping the spatial plots.

    Returns:
    - None: The function creates spatial plots for the AnnData objects.
    """
    def plot_spatial(adata, **kwargs):
        if 'spatial' in adata.obsm:
            sc.pl.spatial(adata, **kwargs)
        else:
            print(f"Spatial coordinates not available for adata. Please add spatial data before plotting.")
    
    adata_dict_fapply(adata_dict, plot_spatial, **kwargs)


