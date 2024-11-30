"""AutoCloud method implementation."""

import numpy as np
import networkx as nx

from itertools import combinations
from collections import OrderedDict

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ..core.distance import minkowski
from ..core.incremental import mean as update_centroid


class _MicroCluster:
    """
    MicroCluster representation for SOStream.

    Parameters
    ----------
    mu : numpy.ndarray, shape=(1, n)
        Center cluster.

    var : float
        Variance or radii of the cluster.

    s : int, optional
        Number of samples used for compute `mu` and `var`.

    Attributes
    ----------
    Same init parameters.

    """

    def __init__(self, centroid=None, radius=0.0, n=1, t0=None):

        self.centroid_ = centroid
        self.radius_ = radius
        self.n_ = n
        self.t0_ = t0
        
        
class _MacroCluster:
    def __init__(self, centroid=None, micro_clusters=None, t0=None):
        

        if centroid is not None and micro_clusters is None:
            micro_clusters = np.array([_MicroCluster(centroid=centroid, t0=t0)])
            
        elif centroid is None and micro_clusters is None:
            micro_clusters = np.empty(shape=0, dtype=object)
        
        self.centroid_ = centroid
        self.micro_clusters_ = micro_clusters
        
    @property
    def n_micro_clusters_(self):
        return self.micro_clusters_.shape[0]
    
    @property
    def micro_centroids_(self):
        return np.array([micro.centroid_.reshape(-1) for micro in self.micro_clusters_])
        
    def new_micro_cluster(self, x, t0=None):

        self.micro_clusters_ = np.r_[self.micro_clusters_, _MicroCluster(centroid=x, t0=t0)]
        
        # Update the macro's centroid
        n_micro_clusters_ = self.n_micro_clusters_
        self.centroid_ = (n_micro_clusters_ * self.centroid_ + x) / (n_micro_clusters_ + 1)
        
    def update(self):
        
        n_micro_clusters_ = self.n_micro_clusters_
        new_centroid = self.centroid_
        new_centroid[:] = 0.0
        
        for micro in self.micro_clusters_:
            new_centroid += micro.centroid_
        
        self.centroid_ = new_centroid / n_micro_clusters_
    

_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'macro_clusters_',
    'n_macro_clusters_',
    't_'
]


class MacroSOStream(ClusterMixin, BaseEstimator):
    """
    Extension of Self Organizing density-based clustering over data Stream to group 
    micro-clusters (MacroSOStream)

    Parameters
    ----------
    m : float
        Sensibility parameter.

    k : float
        Minimum points quantity for create a cluster.

    Attributes
    ----------
    clusters : dict<int, ClusterAutoCloud>
        The current clusters.
            .
            .
            .
    The same init parameters.
    
    References
    ----------
    [1] Oliveira, A. S., de Abreu, R. S., & Guedes, L. A. (2022). 
    Macro SOStream: An Evolving Algorithm to Self Organizing Density-Based 
    Clustering with Micro and Macroclusters. Applied Sciences 2022, Vol. 12, 
    Page 7161, 12(14), 7161. https://doi.org/10.3390/APP12147161

    """

    def __init__(
            self, 
            alpha=0.3,
            min_pts=2,
            merge_threshold=0.1,
            p=1.5,
            merge_macro=True,
            warm_start=False
        ):

        self.alpha = alpha
        self.min_pts = min_pts
        self.merge_threshold = merge_threshold
        self.merge_macro = merge_macro
        self.p = p
        self.warm_start = warm_start
        
    @property
    def all_micro_centroids_(self):
        
        all_micro_centroids_ = np.empty(shape=(1, self.n_features_in_), dtype=np.float64)
        for macro in self.macro_clusters_.values():
            all_micro_centroids_ = np.r_[all_micro_centroids_, macro.micro_centroids_]
            
        return all_micro_centroids_
   
    def fit(self, X, y=None, **kwargs):
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if not self.warm_start:
            self._init(X)
            
        for x in X:
            self.partial_fit(x)
            
        return self

    def partial_fit(self, X, y=None, update_structure=True):
        """
        Process a input and return the assigment values.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New sample.

        Returns
        -------
        isknow : bool
            If the sample it's in cluster transition.

        cluster_assig : int
            Assigment cluster for input.

        """
        
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
            
            if self.n_macro_clusters_ == 0:
                self._new_macro_cluster(X)
            
            else:
                
                macro_cluster_0 = self.macro_clusters_[self.n_macro_clusters_]
                macro_cluster_0.new_micro_cluster(X, t0=self.t_)
           
                if macro_cluster_0.n_micro_clusters_ == self.min_pts:
                    self.initialized_ = True

        else:
            
            new_macro = True
            
            winner_macro_id = None
            winner_micro = None
            
            all_micro_centroids = self.all_micro_centroids_
            
            for j, (macro_id, macro_cluster) in enumerate(self.macro_clusters_.items()):
                
                micro_clusters = macro_cluster.micro_clusters_
                n_micro_clusters = len(micro_clusters)
                
                winner_id, winner_dist = self._find_winner_micro_cluster(X, micro_clusters)
                winner_mc = micro_clusters[winner_id]
                
                winner_nn_ids, dists_win_nn = self._find_winner_neighbors(winner_id, micro_clusters, all_micro_centroids)
                
                if winner_dist <= winner_mc.radius_:
                
                    self._update_micro_clusters(X, winner_id, winner_nn_ids, micro_clusters)
                    
                    overlaps_id, dists_win_nn = self._find_overlap(winner_id, winner_nn_ids, 
                                                                   micro_clusters, dists_win_nn)

                    if len(overlaps_id) > 0:
                        micro_clusters = self._merge_micro_clusters(winner_id, overlaps_id, 
                                                                    micro_clusters, dists_win_nn)
                        
                        macro_cluster.micro_clusters_ = micro_clusters
                    
                    # Update the macro's centroid based on the new micro's centroid
                    macro_cluster.update()
                    
                    winner_macro_id = macro_id
                    winner_micro = winner_mc
                    
                    new_macro = False
                                
                elif winner_dist <= (self.p * winner_mc.radius_):
                    macro_cluster.new_micro_cluster(X, t0=self.t_)
                    
                    winner_macro_id = macro_id
                    winner_micro = macro_cluster.micro_clusters_[-1]  # last micro-cluster
                    
                    new_macro = False
                        
                elif n_micro_clusters < self.min_pts:
                    
                    macro_cluster.new_micro_cluster(X, t0=self.t_)
                    
                    winner_macro_id = macro_id
                    winner_micro = macro_cluster.micro_clusters_[-1]  # last micro-cluster
                    
                    new_macro = False
                    
                if not new_macro:
                    break
                    
                    
            if new_macro:
                self._new_macro_cluster(X)
                
            elif len(self.macro_clusters_) >= 2 and self.merge_macro:
                self._find_and_merge_overlapped_macro(winner_macro_id, winner_micro)
                
                 
        return self
    
    def predict(self, X, **kwargs):
        """
        Assign a cluster for each `x` in `X`.

        Parameters
        ----------
        X : numpy.ndarray, shape=(m, n)
            Set of points.

        Returns
        -------
        labels : numpy.ndarray, shape=(m, )
            Cluster id which each point was assigned.

        """
        
        check_is_fitted(self, _is_fitted_attr)
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        rows, cols = X.shape
        labels = np.zeros(shape=rows, dtype=np.uint16)
        
        if not self.initialized_ or len(self.macro_clusters_) == 0:
            return labels - 1 # labeled as noise

        for k, x in enumerate(X):

            x = x.reshape(1, cols)
            
            win_macro = None
            win_dist = np.inf
            for macro_id, macro in self.macro_clusters_.items():
                _, dist = self._find_winner_micro_cluster(x, macro.micro_clusters_)
                
                if dist < win_dist:
                    win_dist = dist
                    win_macro = macro_id
                
            labels[k] = win_macro
            
        return labels
    
    def _init(self, X):
        
        self.n_features_in_ = X.shape[1]
        self.macro_clusters_ = OrderedDict()
        self.n_macro_clusters_ = 0
        self.t_ = 0
        
        self.is_fitted_ = True
        self.initialized_ = False
        
    def _find_winner_micro_cluster(self, x, micro_clusters):
        
        win_dist = np.inf
        win_micro_cluster_id = None
        
        for mic_id, micro_cluster in enumerate(micro_clusters):
            
            dist = minkowski(x, micro_cluster.centroid_, 
                             p=2, apply_root=True)
            
            if dist < win_dist:
                win_dist = dist
                win_micro_cluster_id = mic_id
                
        return win_micro_cluster_id, win_dist
    
    def _find_winner_neighbors(self, winner_id, micro_clusters, all_micro_centroids):
        
        winner_micro = micro_clusters[winner_id]
        
        n_micro_clusters = len(micro_clusters)
        micro_dists = np.zeros(shape=n_micro_clusters, dtype=np.float64)
       
        for mic_id, micro_cluster in enumerate(micro_clusters):
            
            if mic_id != winner_id:
                micro_dists[mic_id] = minkowski(winner_micro.centroid_, micro_cluster.centroid_,
                                                p=2, apply_root=True)
                
        all_dists = minkowski(winner_micro.centroid_, all_micro_centroids,
                              p=2, apply_root=True).reshape(-1)
            
        idx_micro_sorted= np.argsort(micro_dists)
        idx_all_sorted = np.argsort(all_dists)
        
        micro_dists = micro_dists[idx_micro_sorted]
        all_dists = all_dists[idx_all_sorted]
        
        idx_k = min(n_micro_clusters - 1, self.min_pts - 1)# To avoid error when n_micro_clusters < min_pts due merge operations
        k_dist = all_dists[idx_k]
        winner_micro.radius_ = k_dist
        
        winner_nn = idx_micro_sorted[micro_dists <= k_dist]
        winner_nn = winner_nn[winner_nn != winner_id]
        
        return winner_nn, all_dists
        
    def _find_overlap(self, winner_id, winner_nn_ids, micro_clusters, dists_win_nn):
        
        overlaps = []
        win_mc = micro_clusters[winner_id]
        for win_nn_id in winner_nn_ids:
            
            neighbor_mc = micro_clusters[win_nn_id]
            
            # It is necessary to recalculate the distance because the neighbors' centroids
            # were changed previously (by calling `_update_micro_cluster`)
            dist_win_neighbor = minkowski(win_mc.centroid_, neighbor_mc.centroid_,
                                          p=2, apply_root=True)
            
            if (dist_win_neighbor - (win_mc.radius_ + neighbor_mc.radius_)) < 0:
                overlaps.append(win_nn_id)
            
            # It is updated with the new distance for later use by the `_merge_micro_clusters` method.
            dists_win_nn[win_nn_id] = dist_win_neighbor
            
        return overlaps, dists_win_nn
            
    def _find_and_merge_overlapped_macro(self, winner_macro_id, winner_micro):
        
        winner_macro = self.macro_clusters_[winner_macro_id]
        
        macro_ids = list(self.macro_clusters_.keys())
        for macro_id in macro_ids:
            
            macro_cluster = self.macro_clusters_[macro_id]
            
            if macro_id != winner_macro_id and macro_cluster.n_micro_clusters_ >= self.min_pts:
                
                micro_centroids = macro_cluster.micro_centroids_
                dists = minkowski(winner_micro.centroid_, micro_centroids, p=2, apply_root=True)
                idx_min_dist = np.argmin(dists)
                min_dist = dists[idx_min_dist]
                micro_closer = macro_cluster.micro_clusters_[idx_min_dist]
                
                if (min_dist - (winner_micro.radius_ + micro_closer.radius_)) < self.merge_threshold:
                    
                    n_micro_win_ = winner_macro.n_micro_clusters_
                    n_micro_closer_ = macro_cluster.n_micro_clusters_
                    
                    new_macro_centroid = (n_micro_win_ * winner_macro.centroid_ + 
                                          n_micro_closer_ * macro_cluster.centroid_) / (n_micro_win_ + n_micro_closer_)
                    
                    winner_macro.centroid_ = new_macro_centroid
                    winner_macro.micro_clusters_ = np.r_[winner_macro.micro_clusters_, macro_cluster.micro_clusters_]
                    
                    del self.macro_clusters_[macro_id]
                    
                    break
        
    def _new_macro_cluster(self, x):

        self.n_macro_clusters_ += 1
        self.macro_clusters_[self.n_macro_clusters_] = _MacroCluster(centroid=x, t0=self.t_)
    
    def _update_micro_clusters(self, x, winner_id, winner_nn_ids, micro_clusters):
        
        win_mc = micro_clusters[winner_id]
        
        win_mc.centroid_ = update_centroid(x, win_mc.centroid_, win_mc.n_)
        win_mc.n_ += 1
        win_radius2 = win_mc.radius_ ** 2
        
        for win_n_id in winner_nn_ids:
            
            neighbor_mc = micro_clusters[win_n_id]
            
            dist_win_neighbor = minkowski(win_mc.centroid_, neighbor_mc.centroid_, p=2, apply_root=True)
            influence = np.exp(- (dist_win_neighbor / (2 * win_radius2)))
            
            diff_centroid = win_mc.centroid_ - neighbor_mc.centroid_
            neighbor_mc.centroid_ += self.alpha * influence * diff_centroid
    
    @staticmethod
    def _merge_micro(mi, mj, dist):
        
        wi = mi.n_
        wj = mj.n_
        w_sum = wi + wj
        
        new_centroid = (wi * mi.centroid_ + wj * mj.centroid_) / w_sum
        new_radius = dist + np.maximum(mi.radius_, mj.radius_)
        new_n = w_sum
        
        return new_centroid, new_radius, new_n
        
    def _merge_micro_clusters(self, winner_id, overlaps_id, micro_clusters, dists_win_nn):
        
        to_delete = []
        win_mc = micro_clusters[winner_id]
        for win_n_id in overlaps_id:
        
            neighbor_mc = micro_clusters[win_n_id]
            dist_win_neighbor = dists_win_nn[win_n_id]
            
            if dist_win_neighbor < self.merge_threshold:
                
                new_centroid, new_radius, new_n = self._merge_micro(win_mc, neighbor_mc, dist_win_neighbor)
                
                win_mc.centroid_ = new_centroid
                win_mc.radius_ = new_radius
                win_mc.n_ = new_n
                
                to_delete.append(win_n_id)
                
        # The merged micro-clustrs are excluded
        micro_clusters = np.delete(micro_clusters, to_delete)
                
        return micro_clusters
    
    
    def plot_clusters(self, lw=3, legend=True, ax=None):
        """
        Plot the Group Region and the representative samples.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes when the clusters will be shown. If None (default),
            a new figure is created.

        Returns
        -------
        ellipses : dict<int, matplotlib.patches.Ellipse>
            The clusters representations.

        """
        # if self.dim not in (2, 3):
        #     raise Exception('Available only in 2d and 3d problems.')

        from matplotlib import cm
        from matplotlib import pyplot as plt
        
        from .display import plot_2d_mfgauss_cluster, plot_2d_gauss_cluster

        dim = min(3, self.n_features_in_)
        
        config = {2 : ('rectilinear', plot_2d_mfgauss_cluster)}
        
        cc = config[dim]

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection=cc[0])

        colors = cm.rainbow(np.linspace(0, 1, len(self.macro_clusters_)))
        ells = {}
        
        
        
        for i, (macro_id, macro) in enumerate(self.macro_clusters_.items()):
            
            clabel = f'Cluster {macro_id}'
            lgd = True
            for mc in macro.micro_clusters_:
    
               
                
                # if mc.radius_ > 0:
                ax.scatter(*mc.centroid_.T, color=colors[i], s=30, marker='D')
                ellc = plot_2d_mfgauss_cluster(mc.centroid_,  mc.radius_, 
                                            ls='-', lw=2, color=colors[i], label=clabel if lgd else None, ax=ax)
                    
                lgd = False
                
       
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        
        return ells
    
