import numpy as np

from numpy.linalg import det
from scipy.stats import multivariate_normal

def minkowski(a, b, p=2, norm_factor=1, apply_root=False, keepdims=True):
    
    if p <= 0:
        raise ValueError('`p` must be greater than 0.')
    
    dist = np.sum((np.abs(a - b) / norm_factor) ** p, axis=1, keepdims=keepdims)
    return dist ** (1 / p) if apply_root else dist


def mahalanobis(a, b, inv_cov, apply_root=False):

    diff = a - b
    dist = np.dot(np.dot(diff, inv_cov), diff.T)
    
    return np.sqrt(dist) if apply_root else dist


def mahalanobis_uncorrel(a, b, diag, apply_root=False):
    
    diff = (a - b) ** 2
    dist = np.sum(diff / diag)
        
    return np.sqrt(dist) if apply_root else dist


def bhattacharyya(c_i, c_j, inv_cov_i, inv_cov_j):
    
    # diff = (c_i - c_j)
    # inv_cov = (inv_cov_i + inv_cov_j) / 2
    
    # part_1 = 0.125 * np.dot(np.dot(diff, inv_cov), diff.T)
    # part_2 = 0.5 * np.log((det(inv_cov)) / np.sqrt((det(inv_cov_i)) * det(inv_cov_j)))
    
    # dist = part_1 + part_2
     # return dist
 
    ni = multivariate_normal(c_i[0], np.linalg.pinv(inv_cov_i))
    nj = multivariate_normal(c_j[0], np.linalg.pinv(inv_cov_j))
    
    return np.sqrt(ni.pdf(c_j) * nj.pdf(c_i))
    
