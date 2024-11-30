import numpy as np

from .distance import minkowski
from .loss import relative_squared_error
from .loss import squared_error


def potential(x, X, p=2, dist_metric=None, dkwargs={}):
    
    if dist_metric is None:
        dist = minkowski(x, X, p=p, apply_root=False)
    else:
        dist = dist_metric(x, X, **dkwargs)
    
    w = 1 / X.shape[0]
    return 1 / (1 + w * np.sum(dist))


def cauchy(a, b, alpha=1, p=2, apply_root=False, dist_metric=None, return_dist=False, dkwargs={}):
    
    if dist_metric is None:
        dist = minkowski(a, b, p=p, apply_root=apply_root, **dkwargs)
    else:
        dist = dist_metric(a, b, **dkwargs)

    sml = 1 / (1 + alpha * dist)
    if return_dist:
        return sml, dist
    
    return sml


def imk(a, b, alpha=1, p=2, apply_root=False, dist_metric=None, dkwargs={}):
    
    if dist_metric is None:
        dist = minkowski(a, b, p=p, apply_root=False, **dkwargs)
    else:
        dist = dist_metric(a, b, **dkwargs)
        
    return 1 / np.sqrt(1 + (alpha ** 2) * dist)


def cck(a, b, alpha=1, p=2, apply_root=False, dist_metric=None, dkwargs={}):
    
    if dist_metric is None:
        dist = minkowski(a, b, p=p, apply_root=apply_root, **dkwargs)
    else:
        dist = dist_metric(a, b, **dkwargs)
        
    return 1 / (1 + (alpha ** 2) * dist)


def exp_family(a, b, alpha=1, p=2, apply_root=False, dist_metric=None, dkwargs={}):
    
    if dist_metric is None:
        dist = minkowski(a, b, p=p, apply_root=False, **dkwargs)
    else:
        dist = dist_metric(a, b, **dkwargs)
        
    return np.exp(-alpha * dist)


def estimate_quality(x, xhat, tau, center, return_error=True):
    
    # rserror = squared_error(x, xhat, apply_root=False)
    # rserror_mean = squared_error(center, xhat, apply_root=False)
    
    rserror = relative_squared_error(x, xhat, apply_root=False)
    # rserror_mean = relative_squared_error(center.reshape(1, -1), xhat, apply_root=False)
    
    eq = 1 / (1 + (rserror / tau))
    # eq = 1 / (1 + (rserror))
    
    # eq = np.exp(-rserror)
    # eq = np.exp(-rserror / tau)
    
    if return_error:
        return eq, rserror
    
    return eq


_SML_DICT = {
    'potential': potential,
    'cauchy': cauchy,
    'imk': imk,
    'cck': cck,
    'exp_family': exp_family,
    'estimate_quality': estimate_quality
}


def get_similarity_func_by_name(name):
    
    try:
        return _SML_DICT[name]
    except KeyError:
        raise ValueError("Similarity function not found. The available functions are: %s" % ', '.join(_SML_DICT.keys()))
   
