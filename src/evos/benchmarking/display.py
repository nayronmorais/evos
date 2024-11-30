import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import patheffects

from sklearn.datasets import load_iris, load_diabetes


def parallel_plot(data, target, colormap='viridis', n_ticks=10, precision=3, bar_width=0.1, ax=None):
    
    Y = data.pop(target)
    
    xmin = data.min()
    xmax = data.max()
    
    nsamples, nvars = data.shape
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(False)
    
    x = np.arange(nvars)
    data_norm = (data - xmin)/ (xmax - xmin)
    
    y_min = Y.min()
    y_max = Y.max()
    
    normalize = mpl.colors.Normalize(vmin=y_min, vmax=y_max)
    colors = mpl.colormaps[colormap]
    
    for k in range(nsamples):
        
        sample = data_norm.iloc[k, :]
        y = Y.iloc[k]
        
        
        for i in range(nvars):
            
            color = colors(normalize(y))
            
            xpmin = x[i] - bar_width / 2
            xpmax = x[i] + bar_width / 2
            ax.plot([xpmin, xpmax], [sample[i]] * 2, color=color, lw=2)
            
            if (i + 1) < nvars:
                
                xpmin2 = x[i + 1] - bar_width / 2
                xpmax2 = x[i + 1] + bar_width / 2
                
                ax.plot([xpmax, xpmin2], [sample[i], sample[i + 1]], lw=2, color=color, alpha=0.25)
                # xe = np.c_[xpmin[:, None], x[:, None], xpmax[:, None]].flatten()
        
        # samplee = np.repeat(sample.values, repeats=3)
        
    var_ticks = []
    for i in x:
        
        ticks = np.linspace(xmin.iloc[i], xmax.iloc[i], num=n_ticks)
        
        line = ax.axvline(x=i, lw=2, color='k')
        
        for tick in ticks:
            
            tick_norm = (tick - xmin[i]) / (xmax[i] - xmin[i])
            ax.annotate(np.round(tick, decimals=precision), (i - 0.05, tick_norm), ha='center', va='center', fontsize=10, fontweight='bold')
        
        
        
        
    # for i in range(nvars):
        
       
        
        
        
        
        
        
    ax.set_xticks(x)
    ax.set_xticklabels(data.columns)
    
    ylim = (-0.1, 1.1)
    
    ax.set_ylim(*ylim)
        

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 


data = load_iris(as_frame=True)
X = data.data
Y = data.target

    
n_params = 10
params = dict(
    n_estimators=np.linspace(2, 100, n_params, dtype=int),
    max_depth=np.linspace(2, 30, n_params, dtype=int),
    min_samples_leaf=np.linspace(1, 10, n_params, dtype=int),
    min_samples_split=np.linspace(2, 15, n_params, dtype=int)
    
)

gs = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='roc_auc_ovo_weighted', cv=5, verbose=1)
gs.fit(X, Y)

results = pd.DataFrame.from_dict(gs.cv_results_)
results = results[[f'param_{p}' for p in list(params.keys())] + ['mean_test_score']]

# X['label'] = Y

plt.close('all')
parallel_plot(results, target='mean_test_score', bar_width=0.05)

# pd.plotting.parallel_coordinates(
#     X, 'label'
# ) 