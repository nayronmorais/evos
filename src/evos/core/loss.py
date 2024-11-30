from itertools import combinations

import numpy as np


def squared_error(y, y_pred, norm_factor=1, apply_root=False):
    
    error = np.sum(((y - y_pred) ** 2) / norm_factor, axis=1, keepdims=True)
                   
    if apply_root:
        return np.sqrt(error)
    
    return error
    

def relative_squared_error(y, y_pred, eps=1e-10, apply_root=False):
    
    num = np.sum((y - y_pred) ** 2, axis=1, keepdims=True)
    den = np.sum(y ** 2, axis=1, keepdims=True) + eps
    
    if apply_root:
        
        num = np.sqrt(num)
        den = np.sqrt(den)
    
    return num / den


def mean_squared_error(y, y_pred):
    
    num = y.shape[0]
    error = np.sum((y - y_pred) ** 2) / num
    
    return error


def root_mean_squared_error(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def compute_pair_dist(X, squared=False):
        
    n_samples = X.shape[0]
    Dist = np.zeros(shape=(n_samples, n_samples))
    
    for i, j in combinations(range(n_samples), 2):
        
        xi = X[i]
        xj = X[j]
         
        Dist[i, j] = Dist[j, i] = squared_error(xi[None, :], xj[None, :], squared=squared)
        
    return Dist