
from river.cluster import CluStream as _RiverCluStream
from river.cluster import DBSTREAM as _RiverDBSTREAM
from river.cluster import DenStream as _RiverDenStream
from river.compat import River2SKLClusterer


class CluStream(River2SKLClusterer):
    
    def __init__(
            self,
            n_macro_clusters=5,
            max_micro_clusters=100,
            micro_cluster_r_factor=2,
            time_window=1000,
            time_gap=100,
            seed=None,
            warm_start=False
        ):
        
        self.n_macro_clusters = n_macro_clusters
        self.max_micro_clusters = max_micro_clusters
        self.micro_cluster_r_factor = micro_cluster_r_factor
        self.time_window = time_window
        self.time_gap=time_gap
        self.seed = seed
        self.warm_start = warm_start
        
        self._reset_estimator()
        super().__init__(self.river_estimator)
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self._reset_estimator()
        
        return self
        
    def _reset_estimator(self):
        
        river_estimator = _RiverCluStream(
            n_macro_clusters=self.n_macro_clusters,
            max_micro_clusters=self.max_micro_clusters,
            micro_cluster_r_factor=self.micro_cluster_r_factor,
            time_window=self.time_window,
            time_gap=self.time_gap,
            seed=self.seed
        ) 
        
        self.river_estimator = river_estimator
        
        
class DBSTREAM(River2SKLClusterer):
    
    def __init__(
            self,
            clustering_threshold=1.0,
            fading_factor=0.01,
            cleanup_interval=2,
            intersection_factor=0.3,
            minimum_weight=1.0,
            warm_start=False
        ):
        
        
        self.clustering_threshold = clustering_threshold
        self.fading_factor = fading_factor
        self.cleanup_interval = cleanup_interval
        self.intersection_factor = intersection_factor
        self.minimum_weight=minimum_weight
        self.warm_start = warm_start
        
        self._reset_estimator()
        super().__init__(self.river_estimator)
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self._reset_estimator()
        
        return self
        
    def _reset_estimator(self):
        
        river_estimator = _RiverDBSTREAM(
           clustering_threshold=self.clustering_threshold,
           fading_factor=self.fading_factor,
           cleanup_interval=self.cleanup_interval,
           intersection_factor=self.intersection_factor,
           minimum_weight=self.minimum_weight,
        ) 
        
        self.river_estimator = river_estimator
        
        
class DenStream(River2SKLClusterer):
    
    def __init__(
            self,
            decaying_factor=0.25,
            beta=0.75,
            mu=2,
            epsilon=0.02,
            n_samples_init=1000,
            stream_speed=100,
            warm_start=False
        ):
        
        
        self.decaying_factor = decaying_factor
        self.beta = beta
        self.mu = mu
        self.epsilon = epsilon
        self.n_samples_init=n_samples_init
        self.stream_speed = stream_speed
        self.warm_start = warm_start
        
        self._reset_estimator()
        super().__init__(self.river_estimator)
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        self._reset_estimator()
        
        return self
        
    def _reset_estimator(self):
        
        river_estimator = _RiverDenStream(
            decaying_factor=self.decaying_factor,
            beta=self.beta,
            mu=self.mu,
            epsilon=self.epsilon,
            n_samples_init=self.n_samples_init,
            stream_speed=self.stream_speed,
        ) 
        
        self.river_estimator = river_estimator
