import os

import time
from datetime import datetime

import pandas as pd
import numpy as np
import psutil as psu
import shutil

from collections import OrderedDict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..datasets.datasource import is_collection


class SearchThenPreq:
    
    def __init__(self, save_path=None, clear_results=True, verbose=0, search_kwargs=None):
        
        if search_kwargs is None:
            search_kwargs = {}
        
        if 'cv' not in search_kwargs:
            search_kwargs['cv'] = 5
            
        self.datasource_ = None
        self.save_path = save_path
        self.clear_results = clear_results
        self.verbose = verbose
        self.search_kwargs = search_kwargs
        
        self.methods_ = OrderedDict()
        self.preq_metrics_ = OrderedDict()
        self.preprocessing_ = OrderedDict()
       
    def add_method(self, name, estimator, param_grid):
        self.methods_[name] = (estimator, param_grid)
        
    def add_preq_metric(self, name, metric):
        self.preq_metrics_[name] = metric
        
    def add_preprocessing(self, name, func, **kwargs):
        self.preprocessing_[name] = (func, kwargs)
        
    def clear_preprocessing(self):
        self.preprocessing_.clear()
        
    def set_datasource(self, datasource):
        self.datasource_ = datasource
        
    def _prepare_to_run(self, path):
        
        if self.clear_results:
            shutil.rmtree(path, ignore_errors=True)
        
        os.makedirs(path, exist_ok=True)

    def _save_data_as_dataframe(self, save_epath, filename, data_dict, prefix_name='', timestamp=None):
        
        filename = prefix_name + filename
        if timestamp:
            filename += f'_{timestamp}.csv'
        
        filepath = os.path.join(save_epath, filename)
        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(filepath, index=False)
        
    def apply_data_preprocessing(self, X, Y):
        
        for (preprocess, kwargs) in self.preprocessing_.values():
            X, Y = preprocess(X, Y, **kwargs)
            
        return X, Y

    def run(self):
        
        if self.datasource_ is None:
            raise ValueError("There's no dataset yet, please define it by calling `set_datasource` "
                             "with the proper arguments. ")
        
        datasources = iter(self.datasource_) if is_collection(self.datasource_) else (self.datasource_, )
        timestamp = datetime.now().strftime("%m.%d.%Y %H.%M.%S")

        for datasource in datasources:
            
            if self.verbose > 0:
                print(f"Starting evaluation with datasource `{datasource.filename_}`...")
                
            X, Y = datasource.data, datasource.targets
            X, Y = self.apply_data_preprocessing(X, Y)
                
            save_path = os.path.join(self.save_path, datasource.filename_.split('.')[0])
            self._prepare_to_run(save_path)
            
            for name, (estimator, param_grid) in self.methods_.items():
                
                tstart = time.perf_counter()
                
                if self.verbose > 0:
                    print(f"[STARTED EVALUATION] Method: {name}")
                    
                gs = self.run_grid(X, Y, estimator, param_grid, name)
                self._save_data_as_dataframe(save_path, name, gs.cv_results_, prefix_name='GRID_', timestamp=timestamp)
                
                if len(self.preq_metrics_) > 0:
                    history = self.run_preq(X, Y, estimator, gs.best_params_, name)
                    self._save_data_as_dataframe(save_path, name, history, prefix_name='PREQ_', timestamp=timestamp)
                
                tstop = time.perf_counter()
                if self.verbose > 0:
                    print(f"[FINISHED EVALUATION] Total time: {(tstop - tstart):.3f}s.")
                  
    def run_grid(self, X, Y, estimator, param_grid, name=None):
        
        search_kwargs = self.search_kwargs.copy()
        if search_kwargs['cv'] <= 1:
            indices = np.arange(X.shape[0])
            search_kwargs['cv'] = [(indices, indices)]
            
        gs = RandomizedSearchCV(estimator, param_grid, **search_kwargs)
        gs.fit(X, Y)
        
        if self.verbose > 1:
            
            print('------- GRID SEARCH RESULT ---------')
            print(f'Method: {name}')
            print('Best Params: ', gs.best_params_)
            print('Best Score: ', gs.best_score_)
        
        return gs
    
    def run_preq(self, X, Y, estimator, params, name=None):
        
        estimator.set_params(**params)
        
        history = dict([(name, np.zeros_like(Y, dtype=np.float32)) for name in self.preq_metrics_])
        history['ypred'] =  np.zeros_like(Y)
        
        # Just for initialize the estimator
        estimator.partial_fit(X[0, :].reshape(1, -1))
        
        for t, (x, ytrue) in enumerate(zip(X, Y)):
            
            x = x.reshape(1, -1)
            ypred = estimator.predict(x).item()
            estimator.partial_fit(x)   
            
            for name, metric in self.preq_metrics_.items():
                metric.update(ytrue, ypred)
                history[name][t] = metric.get()
            
            history['ypred'][t] = ypred
            
        return history
