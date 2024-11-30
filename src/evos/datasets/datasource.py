import os
import warnings

import pandas as pd

from scipy.io.arff import loadarff


def is_collection(datsource):
    return hasattr(datsource, '__iter__') 


class TabularData:
    
    def __init__(self, file_path, feature_in_names=None, target_name=None, encoding='utf-8', pandas_kwargs={}):
        
        pandas_kwargs['encoding'] = encoding
        
        self.file_path = file_path
        self.feature_in_names = feature_in_names
        self.target_name = target_name
        self.encoding = encoding
        self.pandas_kwargs = pandas_kwargs
        
        self._load()
        
    def _load(self):
        
        if self.file_path.lower().endswith('arff'):
            
            with open(self.file_path, 'rt', encoding=self.encoding) as file:
                data = loadarff(file)[0]
                fields = list(data.dtype.fields)
                
            data = pd.DataFrame(data, columns=fields)
            
        else:
            data = pd.read_csv(self.file_path, **self.pandas_kwargs)
            fields = data.columns.tolist()
        
        feature_in_names = self.feature_in_names
        if feature_in_names is None:
            feature_in_names = [field for field in fields if field != self.target_name]
        
        self.filename_ = self.file_path.split(os.altsep)[-1]
        self.fields_ = fields
        self.feature_in_names_ = feature_in_names
        self._data_ = data
        
    @property
    def data(self):
        return self._data_[self.feature_in_names_] 
    
    @property
    def targets(self):
        
        if self.target_name is None:
            return
        
        if self.target_name not in self.fields_:
            raise ValueError(f"The target_name named `{self.target_name}` was not found. "  +
                             f"The available fields are {str(self.fields)}")
         
        return self._data_[self.target_name].astype(int)
    

class CollectionTabularData:
    
    def __init__(self, directory, target_names=['class', 'label'], pandas_kwargs={}):
        
        self.directory = directory
        self.target_names = target_names
        self.pandas_kwargs = pandas_kwargs
        
        self._preload()
        
    def _preload(self):
        
        dir_items = os.listdir(self.directory)
        dir_items = [item for item in dir_items if os.path.isfile(os.path.join(self.directory, item))]
        
        self._dir_items_ = dir_items
        self.filename_ = None
        
    def _load(self, filename):
        
        if filename != self.filename_:
            
            self._tbdata_ = None
            
            try:
                file_path = os.path.join(self.directory, filename)
                
                tbdata = TabularData(file_path, target_name=None, pandas_kwargs=self.pandas_kwargs)
                
                for target_name in self.target_names:
                    
                    if target_name in tbdata.fields_:
                        feature_in_names = [field for field in tbdata.fields_ if field != target_name]
                        
                        tbdata.target_name = target_name
                        tbdata.feature_in_names_ = feature_in_names
                        
                        break
                    
                self.tbdata_ = tbdata  
                self.filename_ = filename
                
            except Exception as e:
                error_message = str(e)
                warnings.warn(f"Error loading file `{filename}`, it will be ignored.\nError: \n{error_message}", RuntimeWarning)
            
    @property
    def data(self):
        
        for fname in self._dir_items_:
            self._load(fname)
            
            if self._tbdata_ is None:
                continue
            
            yield self.tbdata_.data
            
    @property
    def targets(self):
        
        for fname in self._dir_items_:
            self._load(fname)
            
            if self._tbdata_ is None:
                continue
            
            yield self.tbdata_.targets
            
    def __iter__(self):
        
        for fname in self._dir_items_:
            self._load(fname)
            
            yield self.tbdata_
            
    
            
        
