# Implementation based on the original code found at https://github.com/cseveriano/evolving_clustering.

import numpy as np
import networkx as nx

from natsort import natsorted

from ..core.distance import minkowski

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted


class ClusterMicroTEDAClus:
    
    def __init__(self, center, variance=0, s=1, density=0, life=1, active=True, changed=True):
        
        self.center = center
        self.variance = variance
        self.s = s
        self.density = density
        self.life = life
        self.active = active
        self.changed = changed
    
    def update_life(self, x, ff=0.01):
        
        if self.variance > 0:
            dist = minkowski(self.center, x, p=2, apply_root=True) # Euclidean distance.
            rt = np.sqrt(self.variance)
            self.life += ((rt - dist) / rt) * ff
            
    def temp_update(self, x):
        
        s = self.s + 1
        w_old = (s - 1) / s
        w_new = 1 / s
        w_new_var = 1 / (s - 1)
        
        center = w_old * self.center +  w_new * x
        variance = w_old * self.variance + (w_new_var * minkowski(center, x, p=2, apply_root=False))
        
        ## Expression used in the original paper
        # variance = w_old * self.variance + (w_new_var * (minkowski(center, x, p=2, scale=True) * 2 / center.size) ** 2)
    
        norm_ecc = MicroTEDAClus.eccentricity(x, center, variance, s, norm=True)
        
        return norm_ecc, center, variance, s
    
    def update(self, x, ff, norm_ecc=None, center=None, variance=None, s=None):
        
        if norm_ecc is None or center is None or variance is None or s is None:
            norm_ecc, center, variance, s = self.temp_update(x)
        
        self.update_life(x, ff)
        
        self.norm_ecc, self.center, self.variance, self.s = norm_ecc, center, variance, s
        self.density = 1 / norm_ecc
        self.changed = True        

    def serialize(self):
        pass
    
    def unserialize(self, serialized_instance):
        pass


_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'micro_clusters_',
    'macro_clusters_',
    'active_macro_clusters_',
    'changed_micro_clusters_',
    'g_',
    'ag_',
    'nmc_',
    't_'
]


class MicroTEDAClus(ClusterMixin, BaseEstimator):
    """
    
    References
    ----------
    
    [1] Maia, J., Severiano Junior, C. A., Gadelha Guimarães, F., Leite De Castro, 
    C., Lemos, P., Camilo, J., Galindo, F., & Cohen, W. (2020). Evolving clustering 
    algorithm based on mixture of typicalities for stream data mining. 
    Future Generation Computer Systems, 106, 672–684. 
    https://doi.org/10.1016/j.future.2020.01.017
    
    """
    
    def __init__(
            self, 
            r0=0.001, 
            decay=100,
            use_r0=True,
            warm_start=False,
        ):
        
        self.r0 = r0
        self.use_r0 = use_r0
        self.decay = decay
        
        
        self.warm_start = warm_start

    @property
    def clusters_(self):
        return dict((mgid, self.micro_clusters_[mcid]) for mgid, mg in enumerate(self.macro_clusters_) for mcid in mg)
    
    @staticmethod
    def eccentricity(x, center, variance, s, norm=True):
        
        lower_bound = 1 / s
        if variance == 0 and s > 1:
            ecc = lower_bound
        else:
            squared_dist = minkowski(center, x, p=2, apply_root=False)
            ecc = lower_bound + (squared_dist / (s * variance))
            
            ## Expression used in the original paper
            # squared_norm_dist = (minkowski(center, x, p=2, scale=True) * 2 / center.size) ** 2
            # ecc = lower_bound + (squared_norm_dist * 2 / (s * variance))
            
        if norm:
            ecc /= 2
            
        return ecc
    
    @staticmethod
    def m(s):
        return 3 / (1 + np.exp(-0.007 * (s - 100)))
    
    @staticmethod
    def threshold(s):
        return (MicroTEDAClus.m(s) ** 2 + 1) / (2 * s)
    
    @staticmethod
    def belongs(norm_ecc, variance, s, r0=0.001, use_r0=True):
        if s <= 2:
            if use_r0:
                return variance < r0
            return True
            
        return norm_ecc <= MicroTEDAClus.threshold(s)
    
    def fit(self, X, y=None, **kwargs):
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if not self.warm_start:
            self._init(X)
            
        for x in X:
            self.partial_fit(x)
            
        return self        
    
    def partial_fit(self, X, y=None, update_structure=True, update_macro_clusters=True, prune_micro_clusters=True):
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        try:
            check_is_fitted(self, _is_fitted_attr)
        except NotFittedError:
            self._init(X)
        
        self.t_ += 1
        
        cid = 1
        if not self.initialized_:
            self._create_new_micro_cluster(X)
            self.initialized_ = True
            
        else:
            
            new_micro_cluster = True
            for mcid, mc in self.micro_clusters_.items():
                
                mc.changed = False
                norm_ecc, center, variance, s = mc.temp_update(X)
                
                if self.belongs(norm_ecc, variance, s, self.r0, self.use_r0):
                    mc.update(X, self.ff_, norm_ecc, center, variance, s)
                    
                    self.changed_micro_clusters_.add(mcid)
                    new_micro_cluster = False

            if new_micro_cluster and update_structure:
                self._create_new_micro_cluster(X)
                
            if prune_micro_clusters and update_structure:
                self._prune_micro_clusters()
                
            if update_macro_clusters:
                self._define_macro_clusters()
                self._define_activations()
                
            cid = self.predict(X).item()
            
        return cid

    def predict(self, X):
        
        check_is_fitted(self, _is_fitted_attr)
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        labels = np.zeros(len(X), dtype=int)
        
        if not self.initialized_ or len(self.active_macro_clusters_) == 0:
           return labels - 1 # labeled as noise
        
        memberships = np.zeros(len(self.active_macro_clusters_), dtype=float)
        
        for k, xk in enumerate(X):
            memberships[:] = 0.0
            for mgid, mg in enumerate(self.active_macro_clusters_):
                active_mc = self._get_active_micro_clusters(mg)
                memberships[mgid] = self.calculate_membership(xk, active_mc)

            labels[k] = np.argmax(memberships) + 1

        return labels
    
    @staticmethod
    def has_intersection(center_i, variance_i, center_j, variance_j):

        dist = minkowski(center_i, center_j, p=2, apply_root=True)
        in_threshold = 2 * (np.sqrt(variance_i) + np.sqrt(variance_j))

        return dist <= in_threshold
    
    @staticmethod
    def isconnected(mcid_i, mcid_j, G):
        return mcid_i in G.neighbors(mcid_j)
    
    @staticmethod
    def calculate_membership(x, active_micro_clusters):
  
        total_density = sum(mc.density for mc in active_micro_clusters.values())

        mb = 0
        for mc in active_micro_clusters.values():
            t = 1 - MicroTEDAClus.eccentricity(x, mc.center, mc.variance, mc.s, norm=True)
            mb += (mc.density / total_density) * t
        return mb

    @staticmethod
    def calculate_micro_membership(x, mc, total_density):

        t = 1 - MicroTEDAClus.eccentricity(x, mc.center, mc.variance, mc.s, norm=True)
        return (mc.density / total_density) * t
    
    def _create_new_micro_cluster(self, x):

        self.nmc_ += 1
        mcid = self.nmc_

        new_mc = ClusterMicroTEDAClus(x)
        self.micro_clusters_[mcid] = new_mc
        self.g_.add_node(mcid)
        
    def _init(self, X):
        
        self.n_features_in_ = X.shape[1]
        self.ff_ = 1 / self.decay
        
        self.micro_clusters_ = {}
        self.macro_clusters_ = []
        self.active_macro_clusters_ = []
        self.changed_micro_clusters_ = set()
        
        self.g_ = nx.Graph()
        self.ag_ = nx.Graph()

        self.nmc_ = 0
        self.t_ = 0
        
        self.is_fitted_ = True
        self.initialized_ = False
        
    def _define_macro_clusters(self):

        for mcid_i in self.changed_micro_clusters_:
            mci = self.micro_clusters_[mcid_i]
            for mcid_j, mcj in self.micro_clusters_.items():
                if mcid_i != mcid_j:
                    
                    edge = mcid_i, mcid_j
                    if MicroTEDAClus.has_intersection(mci.center, mci.variance, mcj.center, mcj.variance):
                        self.g_.add_edge(*edge)
                    elif MicroTEDAClus.isconnected(*edge, self.g_):
                        self.g_.remove_edge(*edge)

        self.macro_clusters_ = natsorted(nx.connected_components(self.g_))
        self.changed_micro_clusters_.clear()
        
    def _define_activations(self):

        self.ag_ = self.g_.copy()

        for mg in self.macro_clusters_:
            num_micro = len(mg)
            total_density = sum(self.micro_clusters_[mcid].density for mcid in mg)
            mean_density = total_density / num_micro

            for mcid in mg:
                mc = self.micro_clusters_[mcid]
                mc.active = mc.s > 2 and mc.density >= mean_density

                if not mc.active:
                    self.ag_.remove_node(mcid)

        self.active_macro_clusters_ = natsorted(nx.connected_components(self.ag_))
    
    def _get_active_micro_clusters(self, mg=None):
        
        if mg is None:
            return dict((mcid, self.micro_clusters_[mcid]) for mg in self.active_macro_clusters_ for mcid in mg)
        
        return dict((mcid, self.micro_clusters_[mcid]) for mcid in mg)
    
    def _prune_micro_clusters(self):
        
        mcids = list(self.micro_clusters_.keys())
        for mcid in mcids:
            mc = self.micro_clusters_[mcid]
            if not mc.active:
                mc.life -= self.ff_
                if mc.life < 0:
                    
                    del self.micro_clusters_[mcid]
                    
                    try:
                        self.changed_micro_clusters_.remove(mcid)
                    except KeyError:
                        pass
                    self.g_.remove_node(mcid)
    
    def plot_micro_clusters(self, X):

        import matplotlib.pyplot as plt
        
        micro_clusters = self._get_active_micro_clusters()
        
        plt.figure()
        ax = plt.gca()
        ax.scatter(*X.T, s=1, color='b')

        for mc in micro_clusters.values():
            circle = plt.Circle(mc.center[0], np.sqrt(mc.variance), color='r', fill=False)
            
            ax.add_artist(circle)
        plt.draw()
        
    def plot_macro_clusters(self, X):
    
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        macro_clusters = self.active_macro_clusters_
        colors = cm.rainbow(np.linspace(0, 1, len(macro_clusters)))
    
        plt.figure()
        ax = plt.gca()
        ax.scatter(*X.T, s=1, color='b')
    
        for mg, c in zip(macro_clusters, colors):
            for mcid in mg:
                mc = self.micro_clusters_[mcid]    
                circle = plt.Circle(mc.center[0], np.sqrt(mc.variance), color=c, fill=False)
                ax.add_artist(circle)
    
        plt.draw()


# %% Toy example
if __name__ == '__main__':
    
    import sys
    sys.path.append('../')
    
    from sklearn.metrics import adjusted_rand_score
    from matplotlib import cm, pyplot as plt
    from sklearn.datasets import make_blobs

    np.random.seed(1)
    
    nsamples = 1000
    nclusters = 5
    center_box = (-200, 200)
    (X, Y) = make_blobs(nsamples, n_features=2, centers=nclusters, cluster_std=10,
                        center_box=center_box, shuffle=False)
    
    # (X, Y) = make_circles(nsamples, shuffle=False, noise=0.1, factor=0.45)
    
    parameters = dict(r0=10, use_r0=True)
    microtedaclus = MicroTEDAClus(**parameters)
    
    # cids = []
    # for x in X:
    #     cid = microtedaclus.partial_fit(x, prune_micro_clusters=True, update_structure=False)
    #     cids.append(cid)
        
    microtedaclus.fit(X)
    cids = microtedaclus.predict(X)
        
    # cids = autocloud.predict(X)
    print('ARS: ', adjusted_rand_score(Y, cids))
    
    microtedaclus.plot_micro_clusters(X)
    microtedaclus.plot_macro_clusters(X)
