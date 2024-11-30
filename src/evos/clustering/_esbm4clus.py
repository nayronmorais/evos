"""Module contains Implementation of the eSBM4Clus."""

import numpy as np

from itertools import combinations
from functools import partial
from collections import OrderedDict

from ..core.similarity import estimate_quality, get_similarity_func_by_name
from ..core.incremental import mean as update_mean
from ..core.loss import squared_error, relative_squared_error
from ..core.utils import stack_upoints
from .display import plot_2d_mfgauss_cluster, plot_3d_gauss_cluster

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted        


class ClustereSBM4Clus:

    def __init__(self, data, sml_func, beta, t0):

        num_samples = data.shape[0]
        
        self.D = D = data
        self.std = np.mean(np.std(data, axis=0))
        self.G, self.invG = eSBM4Clus.compute_matrix_G(D, sml_func, self.std, return_invG=True)
        self.S = np.ones(num_samples)
        
        self.center = np.mean(data, axis=0)
        
        errors = eSBM4Clus.compute_rses(data, self.G, sml_func, self.std)
        
        self.tau = np.mean(errors)
        self.n = num_samples
        self.ns = np.ones(shape=num_samples, dtype=int)
        self.t0 = t0

    def update(self, x, error, a, sml_func, mr, beta, t, gamma=0.1, l_max=50):
        
        a = a.reshape(-1)
        
        idx_sorted = np.argsort(a)[::-1]
        idx_a4 = idx_sorted[:5]
        # self.S[idx_a4] = update_mean(a[idx_a4], self.S[idx_a4], self.ns[idx_a4])
        
        self.S = update_mean(a, self.S, self.n)
        
        beta_adapt = (1 - 2 ** (-0.01 * (t - self.t0))) / 4
        
        if error >= (self.tau * gamma): # and mr >= beta_adapt:
            
            nsamples = self.D.shape[0]
            imax = np.argmax(a)
            imin = np.argmin(a)
            
            # imax = imin
            
            Smax = np.min(self.S)  #self.S[imax]   
            # Smax = self.S[imax]  
            # Smax = np.mean(self.S[idx_a4])

            if nsamples >= l_max:
                
                self.D = np.delete(self.D, imax, axis=0)
                G = np.delete(self.G, imax, axis=0)
                self.G = np.delete(G, imax, axis=1)
                self.S = np.delete(self.S, imax)
                
                a = sml_func(x, self.D, dkwargs={'norm_factor': self.std})
                a = a.reshape(-1)
                    
                nsamples -= 1
                
            G = np.eye(nsamples + 1)
            G[:-1, :-1] = self.G
            G[-1, :-1] = G[:-1, -1] = a
            self.D = np.r_[self.D, x]
            self.G, self.invG = G, np.linalg.pinv(G)
            self.S = np.r_[self.S, Smax]
            # self.ns = np.r_[self.ns, 1]
            
        self.center = update_mean(x, self.center, self.n)
        
        self.tau = update_mean(error, self.tau, self.n)
        # self.ns[idx_a4] += 1
        self.n += 1
        
    def update_from_buffer(self, buffer, sml_func, l_max):
        
        n_buffer = buffer.shape[0]
        n_samples = self.D.shape[0]
        
        if n_samples >= l_max:
            return 
        
        
        for j, x in enumerate(buffer):
            
            x = x.reshape(1, -1)
            a = sml_func(x, self.D, dkwargs={'norm_factor': self.std})
            a = a.reshape(-1)
            
            
            G = np.eye(n_samples + 1)
            G[:-1, :-1] = self.G
            G[-1, :-1] = G[:-1, -1] = a

            imax = np.argmax(a)
            Smax = self.S[imax]
            
            self.G = G
            self.D = np.r_[self.D, x]
            self.S = np.r_[self.S, Smax]
            self.center = update_mean(x, self.center, self.n)
            self.n += 1
            self.ns = np.r_[self.ns, 1]
            
            n_samples += 1
            
            if self.D.shape[0] >= l_max:
                break
            
        self.invG = np.linalg.pinv(G)
        
    def check_updateD(self, error, gamma):
        return error >= (self.tau * gamma)       
        
    def serialize(self):
        pass
    
    def unserialize(self, serialized_instance):
        pass


_is_fitted_attr = [
    'is_fitted_', 
    'clusters_', 
    'n_clusters_', 
    'n_features_in_', 
    'buffer_', 
    'sml_func_',
    't_'
]


class eSBM4Clus(ClusterMixin, BaseEstimator):
    """
    Evolving Similarity-Based Modeling for Online Clustering (eSBM4Clus)".

    Parameters
    ----------
    w : int, optional
        Number of consecutive anomaly points for create a new group. Should be
        at least `dim` + 1. If `None`, `dim` + 1 is used. The default is `None`.

    Attributes
    ----------
    clusters : dict<int, eSBM4ClusPCluster>
        The current clusters.

            .
            .
            .

    The input parameters.

    """

    def __init__(
            self, 
            w=10,
            l_max=200,
            gamma=0.1, 
            beta=0.99, 
            similarity='cauchy',
            alpha=1,
            p=2,
            norm_weights=False,
            warm_start=False,
            
        ):

        self.w = w
        self.gamma = gamma
        self.l_max = l_max
        self.beta = beta
        self.similarity = similarity
        self.alpha = alpha
        self.p = p
        self.norm_weights = norm_weights
        self.warm_start = warm_start
       
    @property
    def models_(self):
        return self.clusters_        

    def fit(self, X, y=None, **kwargs):
        """
        Process a input and return the assigment values.

        Parameters
        ----------
        X : numpy.ndarray, shape=(1, n)
            New sample.
        Y : None
            Just for keep pattern.

        Returns
        -------

        cid : int
           Cluster assigment for the input.

        """
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if not self.warm_start:
            self._init(X)
 
        for x in X:
            self.partial_fit(x)
            
        return self
        
    def partial_fit(self, X, y=None, update_structure=True, return_all=False, **kwargs):
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=False)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        try:
            check_is_fitted(self, _is_fitted_attr)
        except NotFittedError:
            self._init(X)

        self.t_ += 1
        
        cid = 1
        _all = None
        
        if not self.initialized_ and len(self.clusters_) > 0:
            self.initialized_ = True
            
        # if self.t_ % 1000 == 0:
        #     print(f"{self.t_} - CLUSTERS: ", len(self.clusters_), self.get_params())
        
        is_anomaly = True
        if self.initialized_:
            
            (cid, _win_xhat, _win_error, _win_mr, _win_eq, _win_sample_pd, _win_pd, _win_a), _all = self._predict(X, only_cid=False)
            
            # is_anomaly = _win_error > self.clusters_[cid].tau and (not self.use_density or (_win_sample_pd < _win_pd * self.beta))
            is_anomaly = _win_error > self.clusters_[cid].tau and _win_pd <  self.beta
            
            # is_anomaly = _win_mr < self.beta
            if not is_anomaly:
                self.buffer_ = None
                self.clusters_[cid].update(X, _win_error, _win_a, self.sml_func_, _win_mr, self.beta, self.t_, self.gamma, self.l_max)
                
        if is_anomaly and (update_structure or not self.initialized_):
            self.buffer_ = stack_upoints(X, self.buffer_)
            if len(self.buffer_) == self.w:
                cid = self._create_cluster(cid)

        if return_all:
            return cid, _all
        
        return cid
        
    def predict(self, X, return_xhat=False, only_eq=False, **kwargs):
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
        
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, 
                                force_all_finite=True, ensure_2d=True)
        n_samples = X.shape[0]
        labels = np.zeros(shape=n_samples, dtype=np.uint8)
        out = labels
        
        if not self.initialized_ or self.n_clusters_ == 0:
            return labels - 1 # labeled as noise
        
        if return_xhat:
            Xhat = np.zeros_like(X)
            out = labels, Xhat
            
        for k, x in enumerate(X):

            x = x.reshape(1, -1)
            (cid, xhat, error, mr, eq, spd, _, _), _ = self._predict(x, only_cid=False, only_eq=only_eq)
            
            labels[k] = cid
            if return_xhat:
                Xhat[k, :] = xhat
            
        return out
                    
    def _predict(self, x, only_cid=False, only_eq=False, cids=None):
        
        _win_cid = None
        _win_xhat = None
        _win_error = None
        _win_mr = -np.inf
        _win_eq = -np.inf
        _win_sample_pd = -np.inf
        _win_pd = -np.inf
        _win_a = None
        
        _cid = []
        _error = []
        _mr = []
        _eq = []
        _sample_pd = []
        _pd = []
        _a = []
        _xhat = []
        
        if cids is None:
            cids = self.clusters_.keys()
        
        for cid in cids:
            
            cluster = self.clusters_[cid]
            
            xhat, a = self.estimate(x, cluster.D, cluster.invG, self.sml_func_, 
                                    cluster.std, return_a=True, norm_weights=self.norm_weights)
            mr, (eq, sample_pd, pd, error) = self.compute_mr(x, xhat, a, cluster)
            
            _cid.append(cid)
            _mr.append(mr)
            _error.append(error)
            _eq.append(eq)
            _pd.append(pd)
            _sample_pd.append(sample_pd)
            _a.append(a)
            _xhat.append(xhat)
                
            if mr > _win_mr or (only_eq and eq > _win_eq):
                _win_cid = cid
                _win_mr = mr
                _win_error = error
                _win_eq = eq
                _win_sample_pd = sample_pd
                _win_pd = pd
                _win_a = a
                _win_xhat = xhat
        
        _all = _cid, _xhat, _error, _mr, _eq, _sample_pd, _pd, _a
        if only_cid:
            return _win_cid, _all
        
        return (_win_cid, _win_xhat, _win_error, _win_mr, _win_eq, _win_sample_pd, _win_pd, _win_a), _all
    
    def _init(self, X):
        
        self.clusters_ = OrderedDict()
        self.sml_func_ = partial(get_similarity_func_by_name(self.similarity), 
                                 alpha=self.alpha, p=self.p, apply_root=False)
        self.buffer_ = None
        self.n_features_in_ = X.shape[1]
        self.n_clusters_ = 0
        self.t_ = 0
        
        self.initialized_ = False
        self.is_fitted_ = True
        
    def add_model_from_data(self, data):
        
        self.buffer_ = data
        n_cluster_id = self._create_cluster()
        
        return [n_cluster_id]
    
    @staticmethod
    def estimate(x, D, invG, sml_func, std, return_a=False, norm_weights=False):
    
        a = sml_func(x, D, dkwargs={'norm_factor': std}).reshape(-1, 1) 
        w = np.dot(invG, a)
        
        if norm_weights:
            w /= np.sum(np.abs(w))
        xhat = np.dot(D.T, w).T
        
        if return_a:
            return xhat, a
        
        return xhat
    
    @staticmethod
    def compute_potential_density(a, S):
        
        lmax = np.argmax(a)
        max_density = np.max(S)
        
        # pd = S[lmax] / max_density
        sample_pd = np.minimum(1.0, a[lmax] / max_density)
        # pd = np.minimum(1.0, ((S[lmax] * a[lmax])) / max_density)
        # pd = min(1.0, ((a[lmax] * S[lmax]) / (max_density)))
        pd = (a[lmax]) * (S[lmax] / max_density)
        
        return sample_pd, pd
    
    @staticmethod
    def compute_mr(x, xhat, a, cluster):
        
        eq, error = estimate_quality(x, xhat, cluster.tau, cluster.center, return_error=True) 
        sample_pd, pd = eSBM4Clus.compute_potential_density(a, cluster.S)
        
        return min(eq, pd), (eq, sample_pd, pd, error)
        # return (eq * pd), (eq, sample_pd, pd, error)
    
    @staticmethod
    def compute_rses(data, G, sml_func, std):
        
        nrows = data.shape[0]
        idx = np.arange(nrows)
        
        rses = []
        for i in range(nrows):
            
            x = data[i][None, :]
            
            sidx = np.delete(idx, i)
            sdata = data[sidx]
            
            sG = G[sidx, :]
            sG = sG[:, sidx]
            sinvG = np.linalg.pinv(sG)
            
            xhat = eSBM4Clus.estimate(x, sdata, sinvG, sml_func, std, return_a=False)
            # rse = squared_error(x, xhat, apply_root=False)
            rse = relative_squared_error(x, xhat, apply_root=False)
            rses.append(rse.item())
            
        return np.array(rses)
    
    @staticmethod
    def compute_matrix_G(D, sml_func, std, return_invG=True):
        
        nrows = D.shape[0]
        G = np.eye(nrows, dtype=np.float64)
        
        for i, j in combinations(range(nrows), 2):
            
            x1 = D[i][None, :]
            x2 = D[j][None, :]
            
            s = sml_func(x1, x2, dkwargs={'norm_factor': std})
            G[i, j] = G[j, i] = s
            
        if return_invG:
            return G, np.linalg.pinv(G)
        return G
    
    @staticmethod
    def compute_sigma(data):
        
        n_samples, dim = data.shape
        sigma = np.zeros(shape=dim)
        idx = np.arange(n_samples)
        
        for j in range(dim):
            
            dists = []
            for k in range(n_samples):
                
                x_k = data[k, j].reshape(1, 1)
                idx_rest = np.delete(idx, k)
                
                dist = squared_error(x_k, data[idx_rest, j][:, None], squared=False)
                dists.append(np.min(dist))
                
            sigma[j] = np.max(dists)
            
        return sigma
        
        
    ## TODO - Fix decision region
    def plot_decision_region(self, xmin, xmax, X=None, min_mr=0.1, nsamples=100, dcolors=None, 
                             dmarkers=None, ax=None, only_eq=False):
        
        x = np.linspace(xmin, xmax, num=nsamples)
        
        X1, X2 = np.meshgrid(*x.T)
        Z = np.zeros((nsamples, nsamples), dtype=int)
        
        for i in range(nsamples):
            for j in range(nsamples):
                x = np.r_[X1[i, j], X2[i, j]].reshape(1, -1)
                
                (_win_cid, _win_xhat, _win_error, _win_mr, _win_eq, _win_sample_pd, _win_pd, _win_a), _all = self._predict(x)
       
                if _win_mr > min_mr:
                    Z[i, j] = _win_cid
                    
                # cstar = self.predict(x, only_eq=only_eq)
                # Z[i, j] = cstar
                
        from matplotlib import pyplot as plt
        from matplotlib import cm, lines
        from matplotlib.colors import ListedColormap
        import matplotlib as mpl
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
         
        if dcolors is None:
            dcolors = cm.rainbow(np.linspace(0, 1, self.n_clusters_))
            
        if dmarkers is None:
            dmarkers = np.random.choice(list(mpl.lines.Line2D.filled_markers), self.n_clusters_)
            
        cmap = ListedColormap(dcolors)
        cc = ax.contourf(X1, X2, Z, cmap=cmap, alpha=0.45, levels=self.n_clusters_)
        ax.contour(cc, colors='k', linestyles='--', linewidths=1.5)
        
        handles = []
        for i, (id_, cluster) in enumerate(self.clusters_.items()):
            h  = ax.scatter(*cluster.D.T, marker=dmarkers[i], color=dcolors[i+1],
                            s=20, edgecolors='k', lw=0.25, zorder=2)
            handles.append(h)
            
        if X is not None:
            ax.scatter(*X.T, s=20, alpha=0.35, marker='.', color='k')
            
        ax.legend(handles, ['$\\mathrm{D}_{%d}$' % (id_ + 1) for id_ in self.clusters_.keys()])
        
        return ax
            
    def plot_clusters_density(self, xmin, xmax, nsamples=100, cids=None, plot_D=True, with_density=True, show_pd=True, show_eq=True,
                              levels=None, dcolors=None, dmarkers=None, dloc='best', ds=45, 
                              dlfont=10, dlhorizontal=False, dlabelformat='$\\mathrm{D}_{%d}$', add_cbar=True, 
                              cbar_shrink=1, cbar_fraction=0.15, cbar_label='Estimate Quality', 
                              cbar_label_font=10, contour_alpha=0.6, combined=True, ax=None):

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import cm

        def apply_over_grid(X1, X2, cluster):
            n, m = X1.shape
            eqs = np.zeros(shape=(n, m))

            for i in range(n):
                for j in range(m):
                    x = np.array([[X1[i, j], X2[i, j]]])
                    
                    xhat, a = eSBM4Clus.estimate(x, cluster.D, cluster.invG, self.sml_func_, cluster.std, return_a=True)
                    mr, (eq, sample_pd, pd, error) = self.compute_mr(x, xhat, a, cluster)
            
                    # eqs[i, j] = mr if with_density else eq
                    
                    if show_pd and show_eq:
                        eqs[i, j] = mr 
                    elif show_pd:
                        eqs[i, j] = pd
                    else:
                        eqs[i, j] = eq 
                    
                        
            return eqs

        x = np.linspace(xmin, xmax, num=nsamples)
        X1, X2 = np.meshgrid(*x.T)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        if levels is None:
            levels = np.linspace(0.1, 1, 10)
            
        if cids is None:
            cids = self.clusters_.keys()
            
        cn = None
        Z_all = np.zeros_like(X1)
        for cid in cids:
            
            cluster = self.clusters_[cid]

            Z = apply_over_grid(X1, X2, cluster)
            Z_all = np.maximum(Z, Z_all)
            
            if not combined:
                cn = ax.contourf(X1, X2, Z, 
                                  levels=levels, 
                                  cmap='rainbow', 
                                  alpha=contour_alpha, zorder=0)
            
        if combined:
            cn = ax.contourf(X1, X2, Z_all, 
                              levels=levels, 
                              cmap='rainbow', 
                              alpha=contour_alpha, zorder=0)
        
        out = (ax, )
        if cn is not None and add_cbar:
            cb = plt.colorbar(cn, fraction=cbar_fraction, shrink=cbar_shrink, format='%.2f')
            cb.set_label(label=cbar_label, fontsize=cbar_label_font)
            cb.ax.tick_params(labelsize=cbar_label_font-2)
            
            out = (ax, cb)
        
        if dcolors is None:
            dcolors = cm.rainbow(np.linspace(0, 1, self.n_clusters_))
            
        if dmarkers is None:
            dmarkers = np.random.default_rng(10).choice(list(mpl.lines.Line2D.filled_markers), self.n_clusters_)
            
        if plot_D:
            for i, (cid, cluster) in enumerate(self.clusters_.items()):
                ax.scatter(*cluster.D.T, s=ds, marker=dmarkers[i], color=dcolors[i], 
                           edgecolors='k', lw=0.25, label=dlabelformat % cid, zorder=1)
            
            ax.legend(loc=dloc, fontsize=dlfont, markerscale=0.9, labelspacing=0.45, ncol=1 if not dlhorizontal else len(self.clusters_))
        
        return out
    
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

        dim = min(3, self.n_features_in_)
        
        config = {2 : ('rectilinear', plot_2d_mfgauss_cluster), 3 : ('3d', plot_3d_gauss_cluster)}
        
        cc = config[dim]

        if ax is None:
            ax = plt.axes(projection=cc[0])

        colors = cm.rainbow(np.linspace(0, 1, self.n_clusters_))
        ells = {}
        
        plot_clusters_f = cc[1]
        plot_rep_f = ax.scatter if dim == 2 else ax.scatter3D

        for i, (id_, cluster) in enumerate(self.clusters_.items()):

            is_show = False
            
            if first is None:
                is_show = True
                
            elif type(first) is int and id_ <= first:
                is_show = True
                
            elif hasattr(first, '__iter__') and id_ in first:
                is_show = True

            if is_show:
                
                if show_rep_points:
                    plot_rep_f(*cluster.D[:, :dim].T, s=scale_rep_s, 
                               marker='s', color=colors[i], label='$g_{%d}$' % id_)            
                    
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        plt.tight_layout()
        
        return ells
    
    def animate(self, X, nsamples=100, start_delay=1, sleep=0.1, 
                add_lim=0, levels=None, add_cbar=True, figsize=(10, 8), ax=None):
        
        from matplotlib import pyplot as plt
        
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
        ax.axes.spines['top'].set_visible(False)
        ax.axes.spines['right'].set_visible(False)
       
        plt.pause(start_delay)    
        
        xmin = np.inf
        xmax = -np.inf
        cb = None
        
        dmarkers = ['o', 's']
        dcolors = ['red', 'blue']
        ds = 45
                
        for t, x in enumerate(X):
            
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            
            out = self.partial_fit(x, return_all=True)  
            
        
            xmin = np.minimum(xmin, x - add_lim) 
            xmax = np.maximum(xmax, x + add_lim) 
                
            ax.set_xlim(xmin[0], xmax[0])
            ax.set_ylim(xmin[1], xmax[1])
            
            ax.scatter(*X[t].T, color='blue', marker='P', s=12 ** 2, edgecolor='white',
                       linewidths=0.5, label='$x_t$', zorder=2)
            
            msg_title = ''
            
            if self.initialized_:
                
                msg_title = f"$t={t}$, " 
                msg_title += ', '.join(['$| \ \mathrm{D}_{%d} \ | = %d$' % (cid, len(cluster.D)) for cid, cluster in self.clusters_.items()])
                
                (ax, cb) = self.plot_clusters_density(xmin, xmax, nsamples=nsamples, 
                                           dcolors=dcolors, dmarkers=dmarkers, dloc='upper left', ds=ds,
                                           dlfont=10, levels=levels, ax=ax, add_cbar=add_cbar, combined=False,
                                           cbar_fraction=0.05, cbar_shrink=0.65, 
                                           cbar_label='Relação Mínima (MR)', cbar_label_font=11)
                
                if self.buffer_ is not None:
                    ax.scatter(*self.buffer_.T, color='red', marker='*', s=ds, zorder=10)
                    
                
                                
                (cid, (_cid, _xhat, _error, _mr, _eq, _sample_pd, _pd, _a)) = out
                for i, xhat in enumerate(_xhat):
                    ax.scatter(*xhat.T, color='magenta', marker=dmarkers[i], s=12 ** 2, edgecolor='white',
                               linewidths=0.25, label='$\hat{x}_{t%d}$' % (i + 1), zorder=10)
            
            else:
                msg_title += f"Aguardando {t+1}/{self.w}"
                
            ax.set_title(msg_title)
           
            ax.scatter(*X[:t].T, color='k', alpha=0.6, marker='.', s=7 ** 2, edgecolor='white', linewidths=0.25, 
                       zorder=1)
              
            ax.legend()
            
            ax.grid(ls=':', lw=1, alpha=0.3)
            ax.relim()
            plt.draw()
            
            plt.pause(sleep)
        
            if cb is not None:
                cb.remove()
                
            ax.clear()
        
        
    
    def _create_cluster(self, win_cid):
        
        def _create():
            
            self.n_clusters_ += 1
            self.clusters_[self.n_clusters_] = ClustereSBM4Clus(self.buffer_, self.sml_func_, self.beta, self.t_)
            self.buffer_ = None
        
            return self.n_clusters_
        
        if self.buffer_ is None:
            raise ValueError("The buffer is empty.")
        
        if not self.initialized_:
            return _create()
        
        updated = False
        # for cid in self.clusters_:
            
        #     center_buffer = np.mean(self.buffer_, axis=0, keepdims=True)
        #     (cid, xhat, error, mr, eq, spd, _, _), _ = self._predict(center_buffer, only_cid=False, cids=[cid])
            
        #     if mr >= self.beta:
            
        #         cluster_win = self.clusters_[win_cid]
        #         cluster_win.update_from_buffer(self.buffer_, self.sml_func_, self.l_max)
                
        #         updated = True
                
        #         print(f'[{self.t_}] ', 'Merged cluters: ', cid)
        #         break
            
        if not updated:
            return _create()
        self.buffer_ = None
