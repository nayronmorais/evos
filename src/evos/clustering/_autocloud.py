"""AutoCloud method implementation."""

import numpy as np
import networkx as nx

from itertools import combinations
from collections import OrderedDict

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted


class ClusterAutoCloud:
    """
    Cluster representation for AutoCloud.

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

    def __init__(self, mu=None, var=None, s=1):

        self.mu = mu
        self.var = var
        self.s = s

    @classmethod
    def serialize(cls, instance):

        return {
                'mu' : instance.mu.tolist(),
                'var': instance.var,
                's'  : instance.s
            }

    @classmethod
    def unserialize(cls, serialized):

        instance = cls()

        instance.mu = np.array(serialized['mu'])
        instance.var = float(serialized['var'])
        instance.s  = int(serialized['s'])

        return instance

    def temp_update(self, x):
        """
        Compute the attributes update.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New sample.

        Returns
        -------
        mu : numpy.ndarray, shape=(1, n)
            Updated mean.

        var : float
            Updated var.

        s : int
            Updated s.

        """

        s = self.s + 1

        w_old = ((s - 1) / s)

        mu = w_old * self.mu + (1 / s) * x

        diff = x - mu

        var = w_old * self.var + ((1 / s) * np.dot(diff, diff.T).item())

        return mu, var, s

    def update(self, x, mu=None, var=None, s=None):
        """Update the cluster's attributes.

        Use `x` for update if the updated attributes not given.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New sample.

        mu : numpy.ndarray, shape=(1, n), optional
            Updated mean. The default is None.

        var : float, optional
            Updated var. The default is None.

        s : int, optional
            Updated s. The default is None.

        """
        if mu is not None and var is not None and s is not None:
            self.mu = mu
            self.var = var
            self.s = s

        else:
            self.mu, self.var, self.s = self.temp_update(x)


_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'clusters_',
    'n_clusters_',
    'g_',
    't_'
]


class AutoCloud(ClusterMixin, BaseEstimator):
    """
    Autocloud method.

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
            m=2,
            warm_start=False
        ):

        self.m = m
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
        
        # if self.m <= 1:
        #     raise ValueError("`m` must be greater than 1.")
        
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
            self._create_cluster(X)
            self.initialized_ = True
            
        elif self.t_ == 2:
            
            cluster = self.clusters_[1]
            cluster.mu = (cluster.mu + X) / 2
            cluster.var = np.linalg.norm(cluster.mu - X, ord=2) / 2
            cluster.s = 2

        else:

            _win_cid = 1
            _win_ecc = np.inf
            
            belongs = set()
            for cid, cluster in self.clusters_.items():

                mu, var, s = cluster.temp_update(X)
                ecc_norm = AutoCloud.eccentricity(X, mu, var, s, normalized=True)
                b = AutoCloud.belongs(ecc_norm, s, m=self.m)
                
                if b:
                    self.clusters_[cid].update(X, mu=mu, var=var, s=s)                    
                    belongs.add(cid)

                if ecc_norm < _win_ecc:
                    _win_cid = cid
                    _win_ecc = ecc_norm
                        
            cid = _win_cid
            nbel = len(belongs)
            if nbel == 0 and update_structure:
                self._create_cluster(X)     
            elif nbel >= 2:
                
                for cidi, cidj in combinations(belongs, 2):
                    if self.g_.has_edge(cidi, cidj):
                        self.g_.get_edge_data(cidi, cidj)['s'] += 1
                    else:
                        self.g_.add_edge(cidi, cidj, s=1)
    
            toremove = []
            for cidi, cidj in self.g_.edges:                        
                
                try:
                    clusteri, clusterj = self.clusters_[cidi], self.clusters_[cidj]
                    sin = self.g_.get_edge_data(cidi, cidj)['s']
                   
                    if sin > (clusteri.s - sin) or sin > (clusterj.s - sin):
                        
                        self.clusters_[cidi] = AutoCloud.merge(clusteri, clusterj, sin)
                        
                        del self.clusters_[cidj]
                        toremove.append(cidj)
                        
                except KeyError: 
                    # Occurs when there are multiple merges, but here is ignored because 
                    # in the original paper (see [1]) no more than two cluster are combined at a time.

                    pass
                    
            self.g_.remove_nodes_from(toremove)
                                
        return cid
    
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
        
        if not self.initialized_ or len(self.clusters_) == 0:
            return labels - 1 # labeled as noise

        for i, x in enumerate(X):

            x = x.reshape(1, cols)

            win_cid = None
            min_ecc = np.inf
            for cid, cluster in self.clusters_.items():
                ecc_norm = AutoCloud.eccentricity(x, cluster.mu, cluster.var,
                                                  cluster.s, normalized=True)
                
                # print("ECC_norm: ", ecc_norm, ' S: ', cluster.s)

                if min_ecc > ecc_norm:
                    win_cid = cid
                    min_ecc = ecc_norm

            labels[i] = win_cid

        return labels
    
    def _init(self, X):
        
        self.n_features_in_ = X.shape[1]
        self.clusters_ = OrderedDict()
        
        self.g_ = nx.Graph()
        self.t_ = 0
        self.n_clusters_ = 0
        
        self.is_fitted_ = True
        self.initialized_ = False
    
    def _create_cluster(self, x):

        self.n_clusters_ += 1
        self.clusters_[self.n_clusters_] = ClusterAutoCloud(mu=x, var=0, s=1)
        self.g_.add_node(self.n_clusters_)
        

    @staticmethod
    def belongs(ecc_norm, s, m=3):
        """
        Evaluate if is a tipic point.

        Parameters
        ----------
        ecc_norm : float
            Normalized eccentricity.

        s : int
            The number of samples assigned.

        m : float, optional
            Sensibility parameter. The default is 3.

        Returns
        -------
        is_tipic: bool
            If the point is tipic or not.

        """
        return ecc_norm <= ((m ** 2) + 1) / (2 * s)

    @staticmethod
    def eccentricity(x, mu, var, s, normalized=True):
        """
        Compute the eccentricity value.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
           New sample.

        mu : numpy.ndarray, shape=(1, n)
            Current means.

        var : float
            Current variance.

        s : int
            Current number of samples.

        normalized : bool, optional
            If is normalized eccentricity or not. The default is True.

        Returns
        -------
        ecc : float
            The [normalized] eccentricity.

        """
        diff = mu - x
        square_dist = np.dot(diff, diff.T)

        ecc = (1 / s) + (square_dist / (np.maximum(s * var, 1e-20))) # The max operation is intended to avoid indeterminacy.

        if normalized:
            ecc /= 2

        return ecc
    
    @staticmethod
    def merge(clusteri, clusterj, sin):
        """
        Merge two clusters.

        Parameters
        ----------
        a_i : ClusterAutoCloud
            First cluster.
            
        a_j : ClusterAutoCloud
            Second cluster.

        Returns
        -------
        a_ij : ClusterAutoCloud
            Result cluster.

        """
        
        s_i, s_j = clusteri.s, clusterj.s
        
        new_s = s_i + s_j - sin
        new_mu = (s_i * clusteri.mu + s_j * clusterj.mu) / (s_i + s_j) 
        new_var = ((s_i - 1) * clusteri.var + (s_j - 1) * clusterj.var) /  (s_i + s_j - 2)
        
        clusteri.s = new_s
        clusteri.mu = new_mu
        clusteri.var = new_var

        return clusteri
    
    
    def plot_clusters(self, first=None, show_rep_points=True, show_group_region=True, scale_rep_s=6, lw=3, legend=True, ax=None):
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

        colors = cm.rainbow(np.linspace(0, 1, len(self.clusters_)))
        ells = {}
        
        
        for i, (id_, cluster) in enumerate(self.clusters_.items()):

            clabel = f'Cluster {id_}'
            ellc = plot_2d_mfgauss_cluster(cluster.mu, np.ones((1, 2)) * np.sqrt(cluster.s), 
                                        ls='-', lw=1, color=colors[i], label=clabel, ax=ax)
            ells[id_] = ellc
            # labels.append(clabel)    
            
       
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        plt.tight_layout()
        
        return ells
    
  
