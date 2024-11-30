"""AutoCloud method implementation."""

import numpy as np
import networkx as nx

from itertools import combinations
from collections import OrderedDict

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ..core.distance import minkowski
from ..core.incremental import mean as update_centroid


class MicroClusterSOStream:
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

        self.centroid = centroid
        self.radius = radius
        self.n = n
        self.t0 = t0



_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'micro_clusters_',
    'n_micro_clusters_',
    'g_',
    't_'
]


class SOStream(ClusterMixin, BaseEstimator):
    """
    Self Organizing density-based clustering over data Stream (SOStream)

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
    [1] Gomes Bezerra, C., Sielly Jales Costa, B., Affonso Guedes, L., 
    & Parvanov Angelov, P. (2020). An evolving approach to data streams clustering based on 
    typicality and eccentricity data analytics. Information Sciences, 518, 13–28. 
    https://doi.org/10.1016/j.ins.2019.12.022

    """

    def __init__(
            self, 
            alpha=0.3,
            min_pts=2,
            decay=False,
            merge=True,
            weight_decay_rate=0.1,
            interval_decay=100,
            fade_threshold=2,
            merge_threshold=0.1,
            p=2,
            warm_start=False
        ):

        self.alpha = alpha
        self.min_pts = min_pts
        self.merge = merge
        self.decay = decay
        self.weight_decay_rate = weight_decay_rate
        self.interval_decay = interval_decay
        self.fade_threshold = fade_threshold
        self.merge_threshold = merge_threshold
        self.p = p
        self.warm_start = warm_start

    @property
    def n_micro_clusters_(self):
        return self.micro_clusters_.shape[0]
    
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
            self._new_micro_cluster(X)
            self.initialized_ = True

        else:
            
            n_micro_clusters_ = self.n_micro_clusters_
            if n_micro_clusters_ > 0:
                
                winner_id, winner_dist = self._find_winner_micro_cluster(X)
                winner_mc = self.micro_clusters_[winner_id]
                
                if n_micro_clusters_ >= self.min_pts:
                    
                    winner_nn_ids, dists_win_nn = self._find_winner_neighbors(winner_id, winner_mc)
                    
                    recompute_dist = False
                    if winner_dist <= winner_mc.radius:
                       self._update_micro_clusters(X, winner_id, winner_nn_ids)
                       recompute_dist = True
                       
                    else:
                        self._new_micro_cluster(X)
                    
                    if self.merge and len(winner_nn_ids) > 0:
                       
                        overlaps_id, dists_win_nn = self._find_overlap(winner_id, winner_nn_ids, 
                                                                       dists_win_nn, recompute_dist)
                        if len(overlaps_id) > 0:
                            self._merge_micro_clusters(winner_id, overlaps_id, dists_win_nn)
                            
                    if self.decay and (self.t_ % self.interval_decay) == 0:
                        self._fading_all()
                        
                else:
                    self._new_micro_cluster(X)
                    
            else:
                self._new_micro_cluster(X)
                                
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
        labels = np.zeros(shape=rows, dtype=np.int16)
        
        self.define_macro_clusters()
        if not self.initialized_ or len(self.macro_clusters_) == 0:
            return labels - 1 # labeled as noise

        for k, x in enumerate(X):

            x = x.reshape(1, cols)
            win_micro_cluster_id, _ = self._find_winner_micro_cluster(x)
            labels[k] = self.micro2macro(win_micro_cluster_id)
            
        return labels
    
    
    def define_macro_clusters(self):

        mic_ids = [mic_id for mic_id in np.arange(self.n_micro_clusters_) if self.micro_clusters_[mic_id].radius > 0]
        
        self.g_.clear()
        self.g_.add_nodes_from(mic_ids)
        
        for mic_i_id, mic_j_id in combinations(mic_ids, 2):
            
            mc_i = self.micro_clusters_[mic_i_id]
            mc_j = self.micro_clusters_[mic_j_id]
            
            dist = minkowski(mc_i.centroid, mc_j.centroid, p=self.p, apply_root=True)
            if (dist - (mc_i.radius + mc_j.radius)) < 0:
                self.g_.add_edge(mic_i_id, mic_j_id)
                
        self.macro_clusters_ = sorted(nx.connected_components(self.g_))
        self.n_macro_clusters_ = len(self.macro_clusters_)
        
    def micro2macro(self, micro_cluster_id):
        
        if self.n_macro_clusters_ == 0:
            self.define_macro_clusters()
        
        for macro_id, mic_ids in enumerate(self.macro_clusters_):
            if micro_cluster_id in mic_ids:
                return macro_id
            
        return -1
    
    def _init(self, X):
        
        self.n_features_in_ = X.shape[1]
        self.micro_clusters_ = np.empty(shape=0, dtype=object)
        self.macro_clusters_ = []
        
        self.g_ = nx.Graph()
        self.t_ = 0
        self.n_macro_clusters_ = 0
        
        self.is_fitted_ = True
        self.initialized_ = False
        
    def _find_winner_micro_cluster(self, x):
        
        win_dist = np.inf
        winner_id = None
        
        for mic_id, miccluster in enumerate(self.micro_clusters_):
            
            dist = minkowski(x, miccluster.centroid, 
                             p=self.p, apply_root=True)
            
            if dist < win_dist:
                win_dist = dist
                winner_id = mic_id
                
        return winner_id, win_dist
    
    def _find_winner_neighbors(self, winner_id, winner_micro_cluster):
        
        n_micro_clusters = self.n_micro_clusters_
        
        all_dists = np.zeros(shape=n_micro_clusters, dtype=np.float64)
        
        for mic_id, miccluster in enumerate(self.micro_clusters_):
            
            if mic_id != winner_id:
                dist = minkowski(winner_micro_cluster.centroid, miccluster.centroid, 
                                 p=self.p, apply_root=True)
                
                all_dists[mic_id] = dist               
            
        idx_sorted= np.argsort(all_dists)
        all_dists = all_dists[idx_sorted]
      
        k_dist = all_dists[self.min_pts - 1]
        winner_micro_cluster.radius = k_dist
        
        winner_nn = idx_sorted[all_dists <= k_dist]
        winner_nn = winner_nn[winner_nn != winner_id]
        
        return winner_nn, all_dists
        

    def _find_overlap(self, winner_id, winner_nn_ids, dists_win_nn, recompute_dist=False):
        
        overlaps = []
        win_mc = self.micro_clusters_[winner_id]
        for win_n_id in winner_nn_ids:
            
            neighbor_mc = self.micro_clusters_[win_n_id]
            
            if recompute_dist:
                dist_win_neighbor = minkowski(win_mc.centroid, neighbor_mc.centroid, 
                                              p=self.p, apply_root=True)
            else:
                dist_win_neighbor = dists_win_nn[win_n_id]
            
            if (dist_win_neighbor - (win_mc.radius + neighbor_mc.radius)) < 0:
                overlaps.append(win_n_id)
                
            dists_win_nn[win_n_id] = dist_win_neighbor
            
       
        return overlaps, dists_win_nn
    
    def _new_micro_cluster(self, x):
        self.micro_clusters_ = np.r_[ self.micro_clusters_, MicroClusterSOStream(centroid=x, t0=self.t_)]
        
    
    def _update_micro_clusters(self, x, winner_id, winner_nn_ids):
        
        win_mc = self.micro_clusters_[winner_id]
        
        win_mc.centroid = update_centroid(x, win_mc.centroid, win_mc.n)
        win_mc.n += 1
        win_radius2 = win_mc.radius ** 2
        
        for win_n_id in winner_nn_ids:
            
            cluster_neighbor = self.micro_clusters_[win_n_id]
            
            dist_win_neighbor = minkowski(win_mc.centroid, cluster_neighbor.centroid, 
                                          p=self.p, apply_root=True)
            influence = np.exp(- (dist_win_neighbor / (2 * win_radius2)))
            
            diff_centroid = win_mc.centroid - cluster_neighbor.centroid
            cluster_neighbor.centroid += self.alpha * influence * diff_centroid
    
    def _merge_micro_clusters(self, winner_id, overlaps_id, dists_win_nn):
        
        win_mc = self.micro_clusters_[winner_id]
        to_remove = []
        
        for win_n_id in overlaps_id:
        
            neighbor_mc = self.micro_clusters_[win_n_id]
            dist_win_neighbor = dists_win_nn[win_n_id]
            
            if dist_win_neighbor < self.merge_threshold:
                
                wi = win_mc.n
                wj = neighbor_mc.n
                w_sum = wi + wj
                
                new_centroid = (wi * win_mc.centroid + wj * neighbor_mc.centroid) / w_sum
                new_radius = dist_win_neighbor + np.maximum(win_mc.radius, neighbor_mc.radius)
                
                win_mc.centroid = new_centroid
                win_mc.radius = new_radius
                win_mc.n = w_sum
                
                to_remove.append(win_n_id)
                
        self.micro_clusters_ = np.delete(self.micro_clusters_, to_remove)
                
    def _fading_all(self):
        
        to_remove = []
        for mic_id in np.arange(self.n_micro_clusters_):
            
            mc = self.micro_clusters_[mic_id]
            new_n = mc.n * 2 ** (-self.weight_decay_rate * (self.t_ - mc.t0))
            
            if new_n < self.fade_threshold:
                
                try: 
                    self.g_.remove_node(mic_id)
                except nx.exception.NetworkXError: # If `mic_id` is not in the graph.
                    pass
                
            else:
                mc.n = new_n
                
        self.micro_clusters_ = np.delete(self.micro_clusters_, to_remove)
            
    
    def plot_micro_clusters(self, clusters=None, lw=3, legend=True, ax=None):
        """
        Plot the Group Region and the representative samples.

        Parameters
        ----------
        
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
            
        if clusters is None:
            clusters = np.arange(self.n_micro_clusters_)

        colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
        ells = {}
        
        
        for mic_id in clusters:

            mc  = self.micro_clusters_[mic_id]
            clabel = f'Cluster {mic_id}'
           
            
            if mc.radius > 0:
                ellc = plot_2d_mfgauss_cluster(mc.centroid, mc.radius, 
                                            ls='-', lw=2, color=colors[mic_id], label=clabel, ax=ax)
            # ells[id_] = ellc
            # labels.append(clabel)    
            
       
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        
        return ells
    
    def plot_macro_clusters(self, lw=3, legend=True, ax=None):
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
        
        
        
        for i, mic_ids in enumerate(self.macro_clusters_):
            
            clabel = f'Cluster {i}'
            lgd = True
            for id_ in mic_ids:
    
                mc = self.micro_clusters_[id_]
                
                
                # if mc.radius > 0:
                ax.scatter(*mc.centroid.T, color=colors[i], s=30, marker='D')
                ellc = plot_2d_mfgauss_cluster(mc.centroid,  mc.radius, 
                                            ls='-', lw=2, color=colors[i], label=clabel if lgd else None, ax=ax)
                    
                    # ellc2 = plot_2d_gauss_cluster(mc.centroid, np.eye(2) * (1 / mc.radius ** 2), 
                    #                               gamma=self.merge_threshold, ls='--', lw=1, color='blue', ax=ax)
                # ells[id_] = ellc
                # labels.append(clabel)    
                lgd = False
                
       
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        
        return ells
    
