""" MCMSTStream implementation based on the original available at 
https://github.com/senolali/mcmststream """

from collections import OrderedDict

import numpy as np

from ..core.utils import stack_points

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import NotFittedError, check_is_fitted

from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import pdist, squareform, sqeuclidean

_is_fitted_attr = [
    'is_fitted_',
    'n_features_in_',
    'micro_clusters_',
    'macro_clusters_',
    '_counter_micro_',
    '_counter_macro_',
    '_buffer_dtype_',
    '_micro_dtype_',
    '_macro_dtype_',
    'buffer_',
    't_',
    
]


# dist = DistanceMetric(sqeuclidean, dtype=np.float64)

# _dist_params = dict(metric='minkowski', p=2)
_dist_params = dict(metric='euclidean')
_kdtree_params = dict(leaf_size=10, **_dist_params)
_micro_dtype = np.dtype([('ID', np.uint64), ('features', np.float64, (10, ))])


class MCMSTStream(ClusterMixin, BaseEstimator):
    """
    Minimum spanning tree to KD-tree-based micro-clusters (MCMSTStream).

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
            sliding_window_size=10, 
            min_samples=2, 
            min_micros=2, 
            radius=0.1,
            warm_start=False,
        ):
            
        self.sliding_window_size = sliding_window_size
        self.min_samples = min_samples
        self.min_micros = min_micros
        self.radius = radius

        self.warm_start = warm_start

    def fit(self, X, y=None, **kwargs):
            
        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=True)
        
        if not self.warm_start:
            self._init(X)
            
        for x in X:
            self.partial_fit(x, **kwargs)
            
        return self
            
    def partial_fit(self, X, y=None, update_structure=True, **kwargs):
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

        X = self._validate_data(X, dtype='numeric', accept_sparse=False, force_all_finite=True, ensure_2d=False)
        if X.ndim != 1:
            X = X.reshape(-1)
            
        try:
            check_is_fitted(self, _is_fitted_attr)
        except NotFittedError:
            self._init(X)
            
        self.t_ += 1 
        
        # (micro_id, features)
        Xe = (0, X)
        Xe = np.array(Xe, dtype=self._buffer_dtype_) # Conversion to the appropriate numpy dtype 
        self.buffer_ = stack_points(Xe, self.buffer_, 
                                    max_size=self.sliding_window_size,
                                    expand_dim=False, unique=False)
        self._define_micro()
        self._add_to_micro()
        self._define_macro()
        self._add_micro_to_macro()
        self._update_micros()
        self._update_macros()
        self._kill_micros()
        self._kill_macros()
        
        if not self.initialized_:
            if len(self.macro_clusters_) > 0:
                self.initialized_ = True

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

        labels = np.zeros(shape=rows, dtype=np.uint64)
        
        if not self.initialized_:
            return labels - 1 # labeled as noise
        
        try:
            micros = self.micro_clusters_[self.micro_clusters_['macro_id'] != 0]
            center_micros = micros['features']
            kdtree = KDTree(center_micros, **_kdtree_params)
        
            for t, x in enumerate(X):
    
                x = x.reshape(1, cols)
                
                dist, ind = kdtree.query(x, k=1, return_distance=True)
                labels[t] = micros['macro_id'][ind]
        
        except ValueError:
            labels -= 1 # labeled as noise

        return labels
            
    def _init(self, X):
         
        n_features_in_ = X.shape[1] if X.ndim == 2 else X.shape[0]
        buffer_dtype = np.dtype([('micro_id', np.uint64),
                                 ('features', np.float64, (n_features_in_, ))])
                               
        micro_dtype = np.dtype([('micro_id', np.uint64), 
                                ('macro_id', np.uint64),  
                                ('n_samples', np.uint64),
                                ('features', np.float64, (n_features_in_, ))])
        
        macro_dtype = np.dtype([('macro_id', np.uint64), 
                                ('n_micros', np.uint64),  
                                ('edges', object)])
        
        self.n_features_in_ = n_features_in_
        self.micro_clusters_ = np.empty(shape=0, dtype=micro_dtype)
        self.macro_clusters_ = np.empty(shape=0, dtype=macro_dtype)
        self.buffer_ = np.empty(shape=0, dtype=buffer_dtype)
      
        self.t_ = 0
        self._counter_micro_ = 0
        self._counter_macro_ = 0
        
        self._buffer_dtype_ = buffer_dtype
        self._micro_dtype_ = micro_dtype
        self._macro_dtype_ = macro_dtype
        
        self.is_fitted_ = True
        self.initialized_ = False
        

    def _define_micro(self):
        
        idx_unlabeled, = np.nonzero(self.buffer_['micro_id'] == 0)
       
        if idx_unlabeled.size >= self.min_samples:
            
            X = self.buffer_['features'][idx_unlabeled]
            kdtree = KDTree(X, **_kdtree_params)
            for x in X:
                
                nn_points_ind, = kdtree.query_radius(x.reshape(1, -1), r=self.radius)
                n_samples = len(nn_points_ind)
                if n_samples >= self.min_samples:
                    
                    pos_assigned_micros = self.micro_clusters_['macro_id'] != 0
                    center_assigned_micros = self.micro_clusters_['features'][pos_assigned_micros]
                    
                    center_candidate_micro = np.mean(X[nn_points_ind], axis=0)
                    
                    try:
                        kdtree2 = KDTree(center_assigned_micros, **_kdtree_params)
                        dist, _ = kdtree2.query(center_candidate_micro.reshape(1, -1), k=1, return_distance=True)
                    except ValueError: # it occurs when the `center_assigned_micros` is empty
                        dist = np.inf
                        
                    if dist > self.radius * 0.75:
                        
                        self._counter_micro_ += 1
                        
                        # (micro_id, macro_id, n_samples, features)
                        new_micro = (self._counter_micro_, 0, n_samples, center_candidate_micro)
                        new_micro = np.array(new_micro, dtype=self._micro_dtype_)
                        self.micro_clusters_ = np.r_[self.micro_clusters_, new_micro]
                        
                        # Assign the respective data to this new micro-cluster
                        self.buffer_['micro_id'][idx_unlabeled[nn_points_ind]] = self._counter_micro_
                        
                        # Exit the loop
                        break
                    
    def _add_to_micro(self):
        
        if len(self.micro_clusters_) > 1:
            
            idx_non_labeled_data, = np.nonzero(self.buffer_['micro_id'] == 0)
            if idx_non_labeled_data.size > 0:
                
                center_micros = self.micro_clusters_['features']
                kdtree = KDTree(center_micros, **_kdtree_params)
                
                for k in idx_non_labeled_data:
                    data_k = self.buffer_[k]
                    x_k = data_k['features'].reshape(1, -1)
                    
                    dist, ind = kdtree.query(x_k, k=1, return_distance=True)    
                    
                    if dist <= self.radius:
                        self.buffer_['micro_id'][k] = self.micro_clusters_[ind]['micro_id']
                        
    def _define_macro(self):
        
        idx_non_assigned_micros = np.nonzero(self.micro_clusters_['macro_id'] == 0)[0]
        if idx_non_assigned_micros.size >= self.min_micros:
            non_assig_micros = self.micro_clusters_[idx_non_assigned_micros]
            centers = non_assig_micros['features']
            
            Dists = squareform(pdist(centers, **_dist_params))
            edges = self._minimum_spanning_tree(Dists)
            
            
            if len(edges) > 0:
                # Conversion to the global IDs
                out = self._adapt_indices_MST(edges, non_assig_micros)
                edges_by_micro_ids, selected_micro_ids, idx_local = out
                
                n_micros = len(selected_micro_ids)
                total_samples = np.sum(non_assig_micros[idx_local]['n_samples'])
                
                if total_samples >= (self.min_micros * self.min_samples) or n_micros >= self.min_micros:
                    
                    self._counter_macro_ += 1
                    
                    new_macro = (self._counter_macro_, n_micros, edges_by_micro_ids)
                    new_macro = np.array(new_macro, dtype=self._macro_dtype_)
                    self.macro_clusters_ = np.r_[self.macro_clusters_, new_macro]
                    
                    for micro_id in selected_micro_ids:    
                        idx = self.micro_clusters_['micro_id'] == micro_id
                        self.micro_clusters_['macro_id'][idx] = self._counter_macro_
                
    def _add_micro_to_macro(self):
        
        if len(self.macro_clusters_) > 0:
            
            for i, micro in enumerate(self.micro_clusters_):
                
                if micro['macro_id'] == 0 and micro['n_samples'] >= self.min_samples:
                    
                    # Assigned micro-cluster
                    a_micros = self.micro_clusters_['macro_id'] != 0
                    a_micros = self.micro_clusters_[a_micros]
                    
                    if len(a_micros) > 0:
                        center_a_micros = a_micros['features']
                        
                        kdtree = KDTree(center_a_micros, **_kdtree_params)
                        dist, ind = kdtree.query(micro['features'].reshape(1, -1), k=1, return_distance=True)
                        
                        ind = ind.item()
                         
                        if dist <= 2 * self.radius:
                            macro_id = a_micros[ind]['macro_id']
                            self.micro_clusters_['macro_id'][i] = macro_id
                            
                            self._update_macro(macro_id)
                            
                            # Does it just once
                            break

    
    def _update_micros(self):
        
        for i, micro in enumerate(self.micro_clusters_):
            
            idx_samples, = np.nonzero(self.buffer_['micro_id'] == micro['micro_id'])
            n_samples = idx_samples.size
            
            if n_samples > 0:
                self.micro_clusters_['features'][i] = np.mean(self.buffer_['features'][idx_samples], axis=0)
                
            # if micro['macro_id'] != 0:
            #     self.buffer_['macro_id'][idx_samples] = micro['macro_id']
            
            self.micro_clusters_['n_samples'][i] = n_samples
            
    def _update_macros(self):
        
        for macro in self.macro_clusters_:
            self._update_macro(macro['macro_id'])
    
    def _update_macro(self, macro_id):
        
        if macro_id != 0:
            
            idx_macro, = np.nonzero(self.macro_clusters_['macro_id'] == macro_id)
            idx_macro = idx_macro.item()
            
            idx_micros, = np.nonzero(self.micro_clusters_['macro_id'] == macro_id)
            self.micro_clusters_['macro_id'][idx_micros] = 0
            
            micros = self.micro_clusters_[idx_micros]
            center_micros = micros['features']
            
            Dist = squareform(pdist(center_micros, **_dist_params))
            edges = self._minimum_spanning_tree(Dist)
            

            selected_micro_ids = []
            edges_by_micro_ids = np.empty(shape=(0, 2), dtype=np.uint64)
            if len(edges) > 0:
                
                # Conversion to the global IDs
                out = self._adapt_indices_MST(edges, micros)
                edges_by_micro_ids, selected_micro_ids, idx_local = out
                
                self.micro_clusters_['macro_id'][idx_micros[idx_local]] = macro_id
                
                
            self.macro_clusters_['n_micros'][idx_macro] = len(selected_micro_ids)
            self.macro_clusters_['edges'][idx_macro] = np.resize(self.macro_clusters_['edges'][idx_macro], edges_by_micro_ids.shape)
            self.macro_clusters_['edges'][idx_macro] = edges_by_micro_ids
        
    def _kill_micros(self):
        
        to_remove = []
        for i in range(self.micro_clusters_.shape[0]):
            micro = self.micro_clusters_[i]
            macro_id = micro['macro_id']
            
            if micro['n_samples'] == 0:
                
                if macro_id != 0:
                    micro['macro_id'] = 0
                    self._update_macro(macro_id)
                
                to_remove.append(i)
        
        self.micro_clusters_ = np.delete(self.micro_clusters_, to_remove)
    
    def _kill_macros(self):
        
        to_remove = []
        for j in range(self.macro_clusters_.shape[0]):
            
            macro = self.macro_clusters_[j]
            edges = macro['edges']
            micro_ids = np.unique(edges)
            
            global_idx = []
            total_samples = 0
            for micro_id in micro_ids:
                idx_micro, = np.nonzero(self.micro_clusters_['micro_id'] == micro_id)
                micro = self.micro_clusters_[idx_micro]
                
                total_samples += micro['n_samples']
                global_idx.append(idx_micro)
                
            if total_samples < (self.min_micros * self.min_samples) and macro['n_micros'] < self.min_micros:
                self.micro_clusters_['macro_id'][global_idx] = 0
                to_remove.append(j)
                
        self.macro_clusters_ = np.delete(self.macro_clusters_, to_remove)
        
    def _adapt_indices_MST(self, edges, micros):
        
        edges_by_micro_ids = []
        idx_local = []
        for i, j in edges:
            micro_id_i = micros[i]['micro_id']
            micro_id_j = micros[j]['micro_id']
            
            edges_by_micro_ids.append((micro_id_i, micro_id_j))
            idx_local.append(i)
            idx_local.append(j)
        
        edges_by_micro_ids = np.array(edges_by_micro_ids, dtype=np.uint64)
        selected_micro_ids = np.unique(edges_by_micro_ids)
        idx_local = np.unique(idx_local)
        
        return edges_by_micro_ids, selected_micro_ids, idx_local
        
    def _minimum_spanning_tree(self, Weights):
   
       if Weights.shape[0] != Weights.shape[1]:
           raise ValueError("X needs to be square matrix of edge weights")
           
       Weights = Weights.copy()
       n_vertices = Weights.shape[0]
       spanning_edges = []  
      
       # initialize with node 0:                                                                                         
       visited_vertices = [0]                                                                                            
       num_visited = 1
       
       # exclude self connections:
       diag_indices = np.arange(n_vertices)
       Weights[diag_indices, diag_indices] = np.inf
       
       # exclude prohibited connections
       Weights[Weights > 2 * self.radius] = np.inf 
       
       while num_visited != n_vertices:
           new_edge = np.argmin(Weights[visited_vertices], axis=None)
           
           # 2d encoding of new_edge from flat, get correct indices                                                      
           new_edge = divmod(new_edge, n_vertices)
           new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
           
           # if it is True, there's no way to connect to unvisited micro-clusters.
           # This probably means that there is more than one 
           # macro-cluster in this set of micro-clusters.
           if new_edge[0] == new_edge[1]: 
               break
           
            # add edge to tree
           spanning_edges.append(new_edge)
           visited_vertices.append(new_edge[1])
            
           # remove all edges inside current tree
           Weights[visited_vertices, new_edge[1]] = np.inf
           Weights[new_edge[1], visited_vertices] = np.inf                                                                
           
           num_visited += 1
       
       return spanning_edges
    
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
        from mlb4science.display.clusters import plot_2d_gauss_cluster
        
        
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
            plt.pause(delay())
            
            if (t + 1) != len(X): # Keeps the data points of the last iteration.
                points.remove()
                [line.remove() for line in lines]
            
        return ax_plot, ax_scatter
    
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
            
        plt.rcParams['figure.dpi'] = 70
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
        plt.rcParams["figure.figsize"] = (4,4)

        colors = cm.rainbow(np.linspace(0, 1, len(self.micro_clusters_)))
        ells = {}
        
        plot_clusters_f = cc[1]
        plot_rep_f = ax.scatter if dim == 2 else ax.scatter3D

        for i, mic in enumerate(self.micro_clusters_):

            clabel = f'Mic {i}'
            ellc1 = plot_clusters_f(mic['features'][None, :], self.radius, 
                                        ls='-', lw=1, color=colors[i] if mic['macro_id'] != 0 else (0,0,1,1), label=clabel, ax=ax)
          
            
            ells[i] = ellc1
            # labels.append(clabel)   
            
        for edge in self.macro_clusters_['edges']:
            for e in edge:
                i, j = e
                
                idx_i, = np.nonzero(self.micro_clusters_['micro_id'] == i)
                idx_j, = np.nonzero(self.micro_clusters_['micro_id'] == j)
                
                micro_i =  self.micro_clusters_[idx_i.item()]
                micro_j =  self.micro_clusters_[idx_j.item()]
                
                ax.plot([micro_i['features'][0], micro_j['features'][0]], [micro_i['features'][1], micro_j['features'][1]], color='k', lw=1, markersize=500)     
            
      
     
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        # plt.tight_layout()
        plt.grid()
        plt.xlim([-.25,1.25]) 
        plt.ylim([-.25,1.25])  
        
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
            
        plt.rcParams['figure.dpi'] = 70
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
        plt.rcParams["figure.figsize"] = (4,4)

        colors = cm.rainbow(np.linspace(0, 1, len(self.macro_clusters_)))
        ells = {}
        
        plot_clusters_f = cc[1]
        plot_rep_f = ax.scatter if dim == 2 else ax.scatter3D

        for j, macro in enumerate(self.macro_clusters_):
            
            clabel = f"Mic {macro['macro_id']}"
            
            lgd = True
            for  micro_id in np.unique(macro['edges']):
    
                mic = self.micro_clusters_[self.micro_clusters_['micro_id'] == micro_id][0]
                
                
                ellc1 = plot_clusters_f(mic['features'][None, :], self.radius, 
                                            ls='-', lw=1, 
                                            color=colors[j], 
                                            label=clabel if lgd else None, ax=ax)
                
                lgd = False
              
                
                ells[j] = ellc1
                # labels.append(clabel)   
            
        for edge in self.macro_clusters_['edges']:
            for e in edge:
                i, j = e
                
                idx_i, = np.nonzero(self.micro_clusters_['micro_id'] == i)
                idx_j, = np.nonzero(self.micro_clusters_['micro_id'] == j)
                
                micro_i =  self.micro_clusters_[idx_i.item()]
                micro_j =  self.micro_clusters_[idx_j.item()]
                
                ax.plot([micro_i['features'][0], micro_j['features'][0]], [micro_i['features'][1], micro_j['features'][1]], color='k', lw=1, markersize=500)     
            
      
     
        if legend:
            ax.legend(fontsize=10, ncol=5, framealpha=0.6, labelspacing=0.05, columnspacing=1)
        # plt.tight_layout()
        plt.grid()
        plt.xlim([-.25,1.25]) 
        plt.ylim([-.25,1.25])  
        
        return ells
  
