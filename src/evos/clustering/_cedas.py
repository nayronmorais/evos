"""AutoCloud method implementation."""

import numpy as np
import networkx as nx

from itertools import combinations

from natsort import natsorted

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted

from ..core.distance import minkowski
from ..core.incremental import mean as update_center


_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'graph_',
    't_'
]


class CEDAS(ClusterMixin, BaseEstimator):
    """
    Clustering of Evolving Data-streams into Arbitrary Shapes (CEDAS)

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
            min_samples=2,
            radius=0.1,
            decay=1000,
            warm_start=False
        ):

        self.min_samples = min_samples
        self.radius = radius
        self.decay = decay 
        self.warm_start = warm_start

   
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
        if X.ndim == 2:
            X = X.reshape(-1)
            
        try:
            check_is_fitted(self, _is_fitted_attr)
        except NotFittedError:
            self._init(X)
            
        self.t_ += 1 

        if not self.initialized_:
            self._initialization(X)
            self.initialized_ = True
            
        else:

            id_changed = self._update_micros(X)   
            ids_excluded = []
            if self.decay > 1:
                ids_excluded = self._kill_micros()
                
            self._update_graph(id_changed, ids_excluded)
            
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
        
        rows, _ = X.shape
        labels = np.zeros(shape=rows, dtype=np.int16)
        
        self._define_macro_clusters()
        
        if not self.initialized_ or len(self.macro_clusters_) == 0:
            return labels - 1 # labeled as noise
        
        for i, x in enumerate(X):
            
            imin, _  = self._predict(x)
            micro_id = self.micro_clusters_['micro_id'][imin]
            macro_id = self.micro2macro(micro_id, redefine_macro=False)

            labels[i] = macro_id

        return labels
    
    def _predict(self, x):
        
        dists = minkowski(x.reshape(1, -1), self.micro_clusters_['center'], p=2, apply_root=True, keepdims=False)
        imin = np.argmin(dists)
        dist_min = dists[imin]
        
        return imin, (dist_min, dists)
        
    def _define_macro_clusters(self):
        self.macro_clusters_ = natsorted(nx.connected_components(self.graph_))
        
    def micro2macro(self, micro_id, redefine_macro=False):
        
        if len(self.macro_clusters_) == 0 or redefine_macro:
            self._define_macro_clusters()
        
        for j, micro_ids in enumerate(self.macro_clusters_):
            
            if micro_id in micro_ids:
                return j + 1
        
        return -1
    
    def _init(self, X):
        
        self.n_features_in_ = n_features_in_ =  X.shape[1] if X.ndim == 2 else X.shape[0]
        
        micro_dtype = np.dtype([('micro_id', np.uint64),
                                ('n_samples', np.uint64),
                                ('energy', np.float32),
                                ('center', np.float64, (n_features_in_, ))])
        
        self.micro_clusters_ = np.empty(shape=0, dtype=micro_dtype)
        self.macro_clusters_ = []
        
        self._counter_micro_ = 0
        self._micro_dtype_ = micro_dtype
        
        self.graph_ = nx.Graph()

        self.t_ = 0
        self.is_fitted_ = True
        self.initialized_ = False
        
    def _initialization(self, x):
        self._new_micro(x)
    
    def _update_micros(self, x):
        
        id_changed = None
        
        imin, (dist_min, dists) = self._predict(x)
        micro_i = self.micro_clusters_[imin]
        
        if dist_min < self.radius:
            
            micro_i['energy'] = 1.0
            micro_i['n_samples'] += 1
            
            id_changed = micro_i['micro_id']
            
            if dist_min > 0.5 * self.radius: # within shell region
                micro_i['center'] = update_center(x, micro_i['center'], micro_i['n_samples'] - 1)
        
        else:
            self._new_micro(x)
            
        return id_changed
    
    def _kill_micros(self):
        
        self.micro_clusters_['energy'] -= 1 / self.decay
        
        idx_micro_excluded, = np.nonzero(self.micro_clusters_['energy'] < 0)
        
        micro_ids_excluded = self.micro_clusters_['micro_id'][idx_micro_excluded]
        
        self.micro_clusters_ = np.delete(self.micro_clusters_, idx_micro_excluded)
        
        return micro_ids_excluded
    
    def _update_graph(self, id_changed, ids_excluded):
        
        if id_changed is not None:
            
            idx_changed, = np.nonzero(self.micro_clusters_['micro_id'] == id_changed)
            
            micro_changed = self.micro_clusters_[idx_changed.item()]
            micro_changed_id = micro_changed['micro_id']
            
            if micro_changed['n_samples'] >= self.min_samples:
                
                if micro_changed_id not in self.graph_:
                    self.graph_.add_node(micro_changed_id)
            
                dists = minkowski(micro_changed['center'].reshape(1, -1), 
                                  self.micro_clusters_['center'], 
                                  p=2, apply_root=True, keepdims=False)
                
                dists[idx_changed] = np.inf
                is_not_outlier = self.micro_clusters_['n_samples'] >= self.min_samples
                
                connected = (dists < 1.5 * self.radius) & is_not_outlier
                n_connections = np.sum(connected)
    
                if n_connections > 0:
                    micro_ids_connected = self.micro_clusters_['micro_id'][connected]
                    self.graph_.add_edges_from(zip([micro_changed_id] * n_connections, micro_ids_connected))
                
        if len(ids_excluded) > 0:
            self.graph_.remove_nodes_from(ids_excluded)
            
    
    def _new_micro(self, x):

        self._counter_micro_ += 1
        new_micro = np.array((self._counter_micro_, 1, 1.0, x), dtype=self._micro_dtype_)
        self.micro_clusters_ = np.r_[self.micro_clusters_, new_micro]
        
        return self._counter_micro_
    
    
    def plot_micro_clusters(self, first=None, show_rep_points=True, show_group_region=True, scale_rep_s=6, lw=3, legend=True, ax=None):
        """
        Plot the Group Region and the representative samples.

        Parameters
        ----------
        first : int or iterable, optional
            Show only the firsts `first` clusters when is int. Or show specifics
            clusters when `first` is iterable. If None, show all. The default 
            is None.
            
        show_rep_points : bool, optional
            If the representative points will be show. The default is True.
            
        show_group_region : bool, optional
            If the Group Region points will be show. The default is True.
            
        scale_rep_p : float, optional
            The scale of representative point samples. The default is 6.

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
        
        from .display import plot_2d_mfgauss_cluster

        dim = min(3, self.n_features_in_)
        
        config = {2 : ('rectilinear', plot_2d_mfgauss_cluster)}
        
        cc = config[dim]

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection=cc[0])

        colors = cm.rainbow(np.linspace(0, 1, len(self.micro_clusters_)))
        ells = {}
        
        
        for i, micro in enumerate(self.micro_clusters_):

            clabel = f"Cluster {micro['micro_id']}"
            ellc = plot_2d_mfgauss_cluster(micro['center'].reshape(1, -1), self.radius, 
                                        ls='-', lw=1, color=colors[i], label=clabel, ax=ax)
            ells[i] = ellc
            # labels.append(clabel)    
            
       
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        
        return ells
    
    def plot_macro_clusters(self, first=None, show_rep_points=True, show_group_region=True, scale_rep_s=6, lw=3, legend=True, ax=None):
        """
        Plot the Group Region and the representative samples.

        Parameters
        ----------
        first : int or iterable, optional
            Show only the firsts `first` clusters when is int. Or show specifics
            clusters when `first` is iterable. If None, show all. The default 
            is None.
            
        show_rep_points : bool, optional
            If the representative points will be show. The default is True.
            
        show_group_region : bool, optional
            If the Group Region points will be show. The default is True.
            
        scale_rep_p : float, optional
            The scale of representative point samples. The default is 6.

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
        
        from .display import plot_2d_mfgauss_cluster

        dim = min(3, self.n_features_in_)
        
        config = {2 : ('rectilinear', plot_2d_mfgauss_cluster)}
        
        cc = config[dim]

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection=cc[0])

        self._define_macro_clusters()
        
        colors = cm.rainbow(np.linspace(0, 1, len(self.macro_clusters_)))
        ells = {}
        
        for j, micro_ids in enumerate(self.macro_clusters_):
                 
            clabel = f"Cluster {j+1}"
            
            lgd = True
            for micro_id in micro_ids:
                
                micro = self.micro_clusters_[self.micro_clusters_['micro_id'] == micro_id][0]
               
                ellc = plot_2d_mfgauss_cluster(micro['center'].reshape(1, -1), self.radius, 
                                            ls='-', lw=2, color=colors[j], label=clabel if lgd else None, ax=ax)
                ells[j] = ellc
                # labels.append(clabel) 
                
                lgd = False
                
        micros_inactive = self.micro_clusters_[self.micro_clusters_['n_samples'] < self.min_samples]
        for micro in micros_inactive:
            
            ellc = plot_2d_mfgauss_cluster(micro['center'].reshape(1, -1), self.radius, 
                                        ls=':', lw=1, color='k', alpha=0.5, ax=ax)
            
            
       
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        
        return ells
    
  
