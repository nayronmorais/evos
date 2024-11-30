""" Modified version of Online Ellipsoidal Clustering (OEC) method. """

from collections import OrderedDict

import numpy as np
from scipy.stats import chi2

from ..core.distance import mahalanobis
from ..core.utils import stack_upoints

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted


eps = 1e-20


class Tracker:
    """
    tracker class representation.

    Parameters
    ----------
    center : numpy.ndarray, shape=(1, n)
        Initial center of the tracker.

    ff : float
        Forgetting Factor (lambda).

    Attributes
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the tracker.

    inv_cov : numpy.ndarray, shape=(n, n)
        Inverse of the covariance matrix.

    ff : float
        Forgetting Factor (lambda).

    k : int, optional
        Number of samples used for update. The default is 1.


    """

    def __init__(self, center, ff):
        
        self.center = center
        self.cov = np.eye(center.shape[1], dtype=np.float64)
        self.neff = int(3 * (1 / (1 - ff)))
        self.ff = ff
        self.k = 1

    def update(self, x):
        """
        Update tracker's attributes.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New point.

        """
        self.k += 1
        center, cov, ff = self.center, self.cov, self.ff
        
        diff = x - center
        k = min(self.neff, self.k)
        new_center = (ff * center) + ((1 - ff) * x)
        new_cov = ((ff * (k - 1)) / k) * cov + (ff ** 2 / k) * np.dot(diff.T, diff)

        self.center, self.cov = new_center, new_cov

    def unserialize(serialized_instance, **kwargs):
       pass

    def serialize(instance, **kwargs):
        pass


class ClusterOEC:
    """
    Cluster representation for OECPlus.

    Parameters
    ----------
    data : numpy.ndarray, shape=(m, n)
        Data point to use for compute center and inverse of the
        covariance matrix.

    Attributes
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the tracker.

    inv_cov : numpy.ndarray, shape=(n, n)
        Inverse of the covariance matrix.

    k : int
        Number of samples used for update.

    alpha : float
        Sum of weights at time t.

    beta : float
        Square of the sum of weights at time t.

    """

    def __init__(self, data, start=None):

        rows, cols = data.shape

        if start is None:
            
             self.k = rows
             self.alpha = rows
             self.beta = rows
             
             self.center =  np.mean(data, axis=0, keepdims=True)
             
             if rows >= (cols + 1):  # Invertible matrix criteria.
                 self.inv_cov = np.linalg.pinv(np.cov(data, rowvar=False))
             else:
                 self.inv_cov = np.eye(cols, dtype=np.float64)
             
        else:
            
            self.k = 1
            self.alpha = 2
            self.beta = 2 ** 2
            
            center, inv_cov = start
            self.center = center
            self.inv_cov = inv_cov       

    def update(self, x, weight):
        """
        Update cluster's attributes.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New point.

        weight : float
            Weight of `x` to that cluster.

        """
        alpha, beta = self.alpha, self.beta

        new_attrs = OEC.update_weights_attr(weight, alpha, beta)
        new_alpha, new_beta, new_chi, new_delta = new_attrs

        new_center, new_inv_cov = OEC.update_stats(x, weight, new_chi, new_delta,
                                                 new_alpha, self.center, self.inv_cov)

        self.alpha = new_alpha
        self.beta = new_beta
        self.center, self.inv_cov = new_center, new_inv_cov
        self.k += 1

    def unserialize(args, kwargs):
        pass

    def serialize(instance, kwargs):
        pass


_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'gamma_gzone_',
    'tracker_',
    'clusters_',
    'n_clusters_',
    'buffer_',
    'distt_normal_',
    'distt_gzone_',
    't_',
    
]


class OEC(ClusterMixin, BaseEstimator):
    """
    Online Ellipsoidal Clustering (OEC).

    Parameters
    ----------
    stab_period : int
        Stability Period.

    gamma_normal : float, optional
        Confidence interval for normal region. The default is 0.99.

    gamma_gzone : float, optional
        Confidence interval for guard zone. The default is 0.999.

    ff : float, optional
        Forgetting Factor. The default is 0.9.

    Attributes
    ----------
    tracker : tracker class
        tracker instance.

    clusters : dict<int, ClusterOECPlus>
        The current clusters.
            .
            .
            .
    The init parameters.
    
    Notes
    -----
    - This implementation is not exactly the proposed in [1]. In order to avoid
    numeric unstable on the incremental formulas for inverse covariance matrix of the 
    Tracker, we use the covariance matrix (see [2]) instead of the inverse one, since we are
    only interested in the eigenvalues.
    
    References
    ----------
    [1] Moshtaghi, M., Leckie, C., & Bezdek, J. C. (2016). Online clustering of 
    multivariate time-series. 16th SIAM International Conference on Data Mining 2016, 
    SDM 2016, 360–368. https://doi.org/10.1137/1.9781611974348.41.
    
    [2] Moshtaghi, M., Leckie, C., Karunasekera, S., Bezdek, J. C., Rajasegarar, 
    S., & Palaniswami, M. (2011). Incremental elliptical boundary estimation for 
    anomaly detection in Wireless Sensor Networks. Proceedings - IEEE International 
    Conference on Data Mining, ICDM, 467–476. https://doi.org/10.1109/ICDM.2011.80.

    """

    def __init__(
            self, 
            stab_period=10, 
            gamma_normal=0.99, 
            gamma_gzone=None, 
            ff=0.9, 
            c_separation=2,
            new_from_tracker=False,
            warm_start=False,
        ):
            
        self.stab_period = stab_period
        self.gamma_normal = gamma_normal
        self.gamma_gzone = gamma_gzone
        self.ff = ff
        self.new_from_tracker = new_from_tracker
        self.c_separation = c_separation

        self.warm_start = warm_start

    def fit(self, X, y=None, **kwargs):
            
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if not self.warm_start:
            self._init(X)
            
        for x in X:
            self.partial_fit(x, **kwargs)
            
        return self
            
    def partial_fit(self, X, y=None, update_structure=True, return_info=False, **kwargs):
        """
        Process a input and return the assigment values.

        Parameters
        ----------
        X : numpy.ndarray, shape=(1, n)
            New sample.

        Returns
        -------
        isknow : bool
            If the sample it's in cluster transition.

        cluster_assig : int
            Assigment cluster for input.

        """

        global eps
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        try:
            check_is_fitted(self, _is_fitted_attr)
        except NotFittedError:
            self._init(X)
            
        self.t_ += 1 
        
        cid = 1     
        is_anomaly = False
        
        self.tracker_.update(X)
            
        if not self.initialized_:
            
            self.buffer_ = stack_upoints(X, self.buffer_)
            if len(self.buffer_) == self.n_features_in_ + 1:
                self._create_cluster(init=True)
                self.initialized_ = True
            
        else:

            weights = np.zeros(shape=self.n_clusters_, dtype=np.float64)
            dist = np.zeros(shape=self.n_clusters_, dtype=np.float64)
            for i, cluster in  enumerate(self.clusters_.values()):
                weights[i], dist[i] = OEC.compute_weight(X, cluster.center, cluster.inv_cov)

            sum_ = np.sum(weights)
            sum_ = np.max((sum_, eps)) # Avoid indeterminacy.
            weights = weights / sum_
            
            is_anomaly = True
            for i, cluster in  enumerate(self.clusters_.values()):
                
                if cluster.k <= ((self.n_features_in_ + 1) + self.stab_period):
                    cluster.update(X, weights[i])
                    is_anomaly = False
                    self.buffer_ = None

                elif dist[i] <= self.distt_gzone_:
                    cluster.update(X, weights[i])
                    if dist[i] <= self.distt_normal_:
                        is_anomaly = False
                        self.buffer_ = None

            cid = np.argmax(weights) + 1
            
            if is_anomaly and update_structure:
                
                self.buffer_ = stack_upoints(X, self.buffer_)
                n_anomalies = len(self.buffer_)
                
                if n_anomalies == (self.n_features_in_ + 1):

                    is_cseparated = True
                    for i in range(1, self.n_clusters_ + 1):
                        is_cseparated = OEC.is_cseparated(self.clusters_[i], self.tracker_, c=self.c_separation)
                        if not is_cseparated:
                            break
                    
                    if is_cseparated:
                        cid = self._create_cluster()
                        is_anomaly = False
                        
                    # cid = self._create_cluster() 
                    # is_anomaly = False
                      
                        
                    self.buffer_ = None
             
        if return_info:
            return cid, is_anomaly
    
        return self
    
    @property
    def gamma_gzone(self):
        return self.gamma_gzone_
        
    @gamma_gzone.setter
    def gamma_gzone(self, value):
        self.gamma_gzone_ = value
    
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

        labels = np.zeros(shape=rows, dtype=np.uint64)
        
        if not self.initialized_ or self.n_clusters_ == 0:
            return labels - 1 # labeled as noise
        
        if self.n_clusters_ >= 1:
            weights = np.zeros(shape=self.n_clusters_, dtype=np.float64)
    
            for t, x in enumerate(X):
    
                x = x.reshape(1, cols)
                weights[:] = 0.0
    
                for i, cluster in  enumerate(self.clusters_.values()):
                    weights[i] = OEC.compute_weight(x, cluster.center,
                                                    cluster.inv_cov, onlyweight=True)
    
                labels[t] = np.argmax(weights) + 1

        return labels
            
    def _create_cluster(self, init=False):
        
        if self.buffer_ is None:
            raise ValueError("The buffer is empty.")
    
        if init:
            
            center = np.mean(self.buffer_, axis=0, keepdims=True)
            cov = np.cov(self.buffer_, rowvar=False)            
            
            self.tracker_.center = center
            self.tracker_.cov = cov
    
        start = None
        if self.new_from_tracker:
            inv_cov = np.linalg.pinv(self.tracker_.cov)
            start = (self.tracker_.center.copy(), inv_cov)
       
        new_cluster = ClusterOEC(self.buffer_, start=start)
        
        self.n_clusters_ += 1
        self.clusters_[self.n_clusters_] = new_cluster

        self.buffer_ = None
        
        return self.n_clusters_
        
    def _init(self, X):
         
        if len(X) > 1:
            X = X[0][np.newaxis, ...]
            
        n_features_in_ = X.shape[1]
        
        gamma_gzone_ = self.gamma_gzone
        if gamma_gzone_ is None:
            gamma_gzone_ = self.gamma_normal + 0.009
        
        if self.gamma_normal > gamma_gzone_:
            raise ValueError('`gamma_gzone must be equal or greater than `gamma_normal`.')
        
        self.n_features_in_ = n_features_in_
        self.gamma_gzone_ = gamma_gzone_
        self.clusters_ = OrderedDict()
        self.tracker_ = Tracker(X, ff=self.ff)  
        self.buffer_ = X
        
        self.distt_normal_ = chi2.ppf(self.gamma_normal, n_features_in_)
        self.distt_gzone_ = chi2.ppf(self.gamma_gzone_, n_features_in_)
 
        self.t_ = 0
        self.n_clusters_ = 0
        
        self.is_fitted_ = True
        self.initialized_ = False
        
    @staticmethod
    def update_weights_attr(weight, alpha, beta):
        """
        Update the weights for the incremental formulas.

        Parameters
        ----------
        weight : float
            Weight for current input `x`.

        alpha : float
            Current alpha.

        beta : float
            Current beta.

        Returns
        -------
        new_alpha : float
            New alpha.

        new_beta : float
            New beta.

        new_chi : float
            Chi for current input.

        new_delta : float
            Delta for current input.

        """
        

        new_alpha = alpha + weight
        new_beta = beta + (weight ** 2)

        chi_num = beta * ((new_beta ** 2) - new_alpha)
        chi_den = new_beta * ((beta ** 2) - alpha)
        new_chi = chi_num / chi_den

        delta_num = new_beta * ((beta ** 2) - alpha)
        delta_den = beta * weight * (new_beta + weight - 2)
        new_delta = delta_num / delta_den
            
        return new_alpha, new_beta, new_chi, new_delta

    @staticmethod
    def update_stats(x, weight, chi, delta, new_alpha, center, inv_cov):
        """
        Update the cluster's attributes.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            New sample for use to update.

        weight : float
            Weight for `x`.

        chi : float
            Chi for chi.

        delta : float
            Delta for chi.

        new_alpha : float
            Current alpha.

        center : numpy.ndarray, shape=(1, n)
            Current center.

        inv_cov : numpy.ndarray, shape=(n, n)
            Current inverse of the covariance matrix.

        Returns
        -------
        new_center : numpy.ndarray, shape=(1, n)
            Updated center.

        new_inv_cov : numpy.ndarray, shape=(n, n)
            Updated inverse of the covariance matrix.

        """
        
        diff = x - center
        numerator = np.dot(inv_cov,
                           np.dot(diff.T,
                                  np.dot(diff, inv_cov)))

        denominator = delta + np.dot(diff, np.dot(inv_cov, diff.T))

        new_inv_cov = chi * (inv_cov - (numerator / denominator))
        new_center = center + (weight / new_alpha) * diff

        return new_center, new_inv_cov

    @staticmethod
    def compute_weight(x, center, inv_cov, onlyweight=False):
        """
        Compute the weight for `x`.

        Parameters
        ----------
        x : numpy.ndarray, shape=(1, n)
            M-dimensional sample.

        center : numpy.ndarray, shape=(1, n)
            Center of the cluster.

        inv_cov : numpy.ndarray, shape=(n, n)
            Inverse of the covariance matrix.

        onlyweight : bool, optional
            If the return should be only the weight or not.
            The default is False.

        Returns
        -------
        weight : float
            Weight for `x`.

        """
        
        dist = mahalanobis(x, center, inv_cov=inv_cov, apply_root=False).item()
        try:
            with np.errstate(all='raise'):
                weight = np.exp(-0.5 * dist)
        except FloatingPointError:  # When the dist is too huge.
            global eps
            weight = eps
            
        weight = np.clip(weight, eps, 1.0) # Avoid numeric issues
        
        if onlyweight:
            return weight
        return weight, dist

    @staticmethod
    def is_cseparated(cluster, tracker, c=2):
        """
        Check if the `cluster` is c-separated from the `tracker`.

        Parameters
        ----------
        cluster : ClusterOEC class
            A cluster instance.

        tracker : tracker class
            A tracker instance.

        c : int, optional
            The c separated level. The default is 2.

        Returns
        -------
        is_cseparated : bool
            If the cluster is c-separated of the tracker.

        """
        eingvalues_c, _ = np.linalg.eigh(cluster.inv_cov)
        eingvalues_t, _ = np.linalg.eigh(tracker.cov)

        max_eingvalue_c = 1 / np.min(eingvalues_c[np.nonzero(eingvalues_c)])    
        max_eingvalue_t = np.max(eingvalues_t)

        diff = cluster.center - tracker.center
        dist = np.sqrt(np.dot(diff, diff.T))  # Euclidean distance.

        dim = tracker.center.shape[1]
        cscore = c * np.sqrt(dim * max(max_eingvalue_c, max_eingvalue_t))

        return dist >= cscore
    
    def animate(self, X, start_delay=0, delay=0.5, stop=None, ax=None, figsize=(10, 9), 
                c_colors=20, fmt_clabel=None, series_label=None, series_colors=None, series_window_view=500):
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if X.shape[1] != 2:
            raise ValueError("The data dimension must be 2.")
            
        if fmt_clabel is None:
            fmt_clabel = 'Cluster {cid}'
            
        if series_label is None:
            series_label = ['$x_1$', '$x_2$']
            
        if series_colors is None:
            series_colors = ['blue', 'red']
        
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from matplotlib import cm
        from .display import plot_2d_gauss_cluster
        
        
        self._init(X)
        
        colors = cm.jet(np.linspace(0, 1, c_colors)); np.random.shuffle(colors)
        
        if ax is None:
        
            fig = plt.figure(figsize=figsize)
            gs = mpl.gridspec.GridSpec(100, 1, figure=fig, bottom=0.1, top=0.95, right=0.95, left=0.1)
           
            ax_scatter = fig.add_subplot(gs[:66])
            ax_plot = fig.add_subplot(gs[78:])
            
            ax_plot.set_title("Temporal Perspective", fontsize=12, fontweight=500)
            ax_plot.set_xlabel("Time (t)", fontsize=12)
            ax_plot.set_ylabel("Signal", fontsize=12)
            
            ax_plot.grid(ls=':', lw=0.5, alpha=0.4)
            ax_plot.axes.spines['top'].set_visible(False)
            ax_plot.axes.spines['right'].set_visible(False)
            ax_plot.set_prop_cycle(color=series_colors)
            
            ax_scatter.set_title("Spatial Perspective", fontsize=12, fontweight=500)
            plt.setp(ax_scatter.get_xticklabels(), color=series_colors[0])
            plt.setp(ax_scatter.get_yticklabels(), color=series_colors[1])
            
            ax_scatter.grid(ls=':', lw=0.5, alpha=0.4)
            ax_scatter.axes.spines['top'].set_visible(False)
            ax_scatter.axes.spines['right'].set_visible(False)
            
        else:
            ax_plot, ax_scatter = ax
            
        def on_close(event):
            event.canvas.figure.axes[0].has_been_closed = True
            
        fig = plt.gcf()
        fig.axes[0].has_been_closed = False
        fig.canvas.mpl_connect('close_event', on_close)
        
        ax_scatter.set_xlabel(series_label[0], fontsize=12, color=series_colors[0])
        ax_scatter.set_ylabel(series_label[1], fontsize=12, color=series_colors[1])
        
            
        clusters_artist_normal = {}
        clusters_artist_gzone = {}
        tracker_artist = None
        
        
        for t, x in enumerate(X):
            
            if fig.axes[0].has_been_closed or (stop is not None and stop()):
                break
            
            self.partial_fit(x)
            
            xlims = max(0, t - series_window_view) - 1, t + 20
            ax_plot.set_xlim(*xlims)
            
            
            for i, (cid, cluster) in enumerate(self.clusters_.items()):
                
                if tracker_artist is not None:
                    tracker_artist.remove()
                    
                tracker_artist = plot_2d_gauss_cluster(self.tracker_.center,self.tracker_.cov, gamma=self.gamma_normal, 
                                                ls='--', lw=3, color='k', label="Tracker", is_inv_cov=False, ax=ax_scatter)
                
                clabel = fmt_clabel.format(cid=cid)
                if cid not in clusters_artist_normal:
                    
                    new_c_artist_normal = plot_2d_gauss_cluster(cluster.center, cluster.inv_cov, gamma=self.gamma_normal, 
                                                ls='-', lw=2, color=colors[i], label=clabel, ax=ax_scatter)
                    
                    new_c_artist_gzone = plot_2d_gauss_cluster(cluster.center, cluster.inv_cov, gamma=self.gamma_gzone_, 
                                                ls='--', color=colors[i], lw=1, ax=ax_scatter)
                    
                    clusters_artist_normal[cid] = new_c_artist_normal
                    clusters_artist_gzone[cid] = new_c_artist_gzone
                    
                else:
                    
                    clusters_artist_normal[cid].remove()
                    clusters_artist_gzone[cid].remove()
                    
                    new_c_artist_normal = plot_2d_gauss_cluster(cluster.center, cluster.inv_cov, gamma=self.gamma_normal, 
                                                ls='-', lw=2, color=colors[i], label=clabel, ax=ax_scatter)
                    
                    new_c_artist_gzone = plot_2d_gauss_cluster(cluster.center, cluster.inv_cov, gamma=self.gamma_gzone_, 
                                                ls='--', color=colors[i], lw=1, ax=ax_scatter)
                    
                    clusters_artist_normal[cid] = new_c_artist_normal
                    clusters_artist_gzone[cid] = new_c_artist_gzone
                    
               
            lines = ax_plot.plot(X[:t], lw=2, markevery=[-1], marker='o')
            points = ax_scatter.scatter(*X[:t].T, marker='o', s=30, color='gray', linewidths=0.5, edgecolor='black')
            
            ax_scatter.legend()
            ax_plot.legend(series_label)
            
            
            
            plt.draw()
            plt.pause(delay)
            
            if (t + 1) != len(X): # Keeps the data points of the last iteration.
                points.remove()
                [line.remove() for line in lines]
            
        return ax_plot, ax_scatter
    
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
        
        from .display import plot_2d_gauss_cluster, plot_3d_gauss_cluster

        dim = min(3, self.n_features_in_)
        
        config = {2 : ('rectilinear', plot_2d_gauss_cluster), 3 : ('3d', plot_3d_gauss_cluster)}
        
        cc = config[dim]

        if ax is None:
            fig = plt.figure()
            ax = plt.axes(projection=cc[0])

        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters_))
        ells = {}
        
        plot_clusters_f = cc[1]
        plot_rep_f = ax.scatter if dim == 2 else ax.scatter3D

        for i, (id_, cluster) in enumerate(self.clusters_.items()):

            clabel = f'Cluster {id_}'
            ellc1 = plot_clusters_f(cluster.center, cluster.inv_cov, gamma=self.gamma_normal,
                                        ls='-', lw=1, color=colors[id_ - 1], label=clabel, ax=ax)
            
            ellc2 = plot_clusters_f(cluster.center, cluster.inv_cov, gamma=self.gamma_gzone_,
                                        ls=':', lw=1, color=colors[id_ - 1], ax=ax)
            
            ells[id_] = ellc1
            ells[f'{id_}_'] = ellc2
            # labels.append(clabel)    
            
        tlabel = 'Tracker'
        ellt = plot_clusters_f(self.tracker_.center, np.linalg.pinv(self.tracker_.cov), gamma=self.gamma_gzone_, 
                                      ls='--', color='darkmagenta', label=tlabel, ax=ax)
        ells[-1] = ellt
         
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        # plt.tight_layout()
        
        return ells
  
