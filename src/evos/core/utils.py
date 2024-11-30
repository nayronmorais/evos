import os
import time

import numpy as np

from natsort import natsorted
from collections import OrderedDict


def make_dirs(*paths):
    fpath = os.path.join(*paths)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def stack_points(new_point, data, max_size=-1, expand_dim=True, unique=False):
    
    if isinstance(new_point, list):
        new_point = np.array(new_point)
    
    if new_point.ndim == 1 and expand_dim:
        new_point = new_point[None, ...]
    
    if data is None:
        return new_point
    
    if unique:  # Avoid duplicate samples.
        
        over_axis = 1 if new_point.ndim > 1 else 0
        if not np.any(np.all(new_point == data, axis=over_axis)):
            data = np.r_[data, new_point]
    else:
        data = np.r_[data, new_point]
        
    if max_size > 0:
        data = data[-max_size:]
        
    return data


def stack_upoints(new_point, data, max_size=0, expand_dim=True):
    return stack_points(new_point, data, max_size=max_size, expand_dim=expand_dim, unique=True)


def check_dir(path):
    
    if not os.path.exists(path):
        raise ValueError('The path does not exist.')
        
    if not os.path.isdir(path):
        raise ValueError('The path is not a directory.')
        
        
def check_file(filepath, sep=None, usecols=None):
    
    if not os.path.exists(filepath):
        raise ValueError('The path does not exist.')
        
    if not os.path.isfile(filepath):
        raise ValueError('The path is not a file.')
        
def check_model(model):
    pass

def check_dataprovider(dataprovider):
    pass

def check_metric(metric):
    pass

def check_preprocess(func):
    pass

def adjust_modelname(modelname: str, parameters: dict) -> str:
    joined_params = '_'.join([f"{p}.{str(v).replace('.', '')}" for p, v in parameters.items()])
    return f"{modelname}-{joined_params}"

def build_exec_folder_name():
    return time.strftime('%d.%m.%Y %H.%M.%S', time.localtime())     

def map_dirfiles(rootdir, sort=True) -> OrderedDict:
    
    def listfiles(path):
        fpath = os.path.join(rootdir, path)
        files = [f for f in os.listdir(fpath) if os.path.isfile(os.path.join(fpath, f))]
        if sort:
            files = natsorted(files)
        return (path, files)
    
    mappedfiles = OrderedDict()
    files = [f for f in os.listdir(rootdir) if os.path.isfile(os.path.join(rootdir, f))]
    if sort:
        files = natsorted(files)
    mappedfiles[''] = files
    
    basedirs = [d for d in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, d))]
    if len(basedirs) > 0:
        if sort:
            basedirs = natsorted(basedirs)
        mappedfiles.update(map(listfiles, basedirs))
        
    return mappedfiles


def build_lagged_matrix(X, lags=3):
    """
    Build a lagged variable for each ones in X.

    Parameters
    ----------
    X : numpy.ndarray, shape=(m, n)
        Reference points.

    lags : int, optional
        The desired lags. The default is 3.

    Returns
    -------
    lagged_matrix : numpy.ndarray, shape=(m, n * (1 + l))
        Matrix com lagged variables.

    """
    if lags == 0:
        return X

    def apply_column(row_idxs, data_row):

        return data_row[row_idxs]

    rows, cols = X.shape
    new_cols = cols + cols * lags

    lagged_mat = np.zeros(shape=(rows, new_cols), dtype=np.float64)
    lagged_mat[:, :cols] = X
    lagged_mat[0, cols:] = 0

    # Building matrix with lagged indexes.
    rows_idxs = np.arange(rows)[:, None]
    lag_vec = list(range(1, lags + 1))
    lag_mat_idxs = np.tile(lag_vec, reps=(rows, 1))
    lag_mat_idxs = rows_idxs - lag_mat_idxs
    lag_mat_idxs_zeros = np.triu_indices(lags)

    l = cols

    for i in range(cols):

        row = lagged_mat[:, i]

        new_lagged_rows = np.apply_along_axis(apply_column, 0, lag_mat_idxs, row)
        new_lagged_rows[lag_mat_idxs_zeros] = 0
        lagged_mat[:, l:l + lags] = new_lagged_rows

        l += lags

    return lagged_mat


def add_lagged_vars(x, matrix, lags=3):
    """
    Build a sample with lagged variables.

    Parameters
    ----------
    x : numpy.ndarray, shape=(1, n)
        Point to add lagged vars.

    matrix : numpy.ndarray, shape=(m, n)
        Reference points.

    lags : int, optional
        The desired lags. The default is 3.

    Returns
    -------
    lagged_sample : numpy.ndarray, shape=(1, n * (1 + l))
        Sample com lagged variables.

    """
    if lags == 0:
        return x

    cols = x.shape[1]
    new_cols = cols + cols * lags

    new_x = np.zeros(shape=(1, new_cols), dtype=np.float64)
    new_x[:, :cols] = x

    l = cols

    lag_idxs = np.array([-i for i in range(1, lags + 1)])[None, :]
    apply = lambda idx, r: r[idx]

    for i in range(cols):

        column = matrix[:, i]

        new_vars = np.apply_along_axis(apply, 0, lag_idxs, column)
        new_x[0, l: l + lags] = new_vars

        l += lags

    return new_x
