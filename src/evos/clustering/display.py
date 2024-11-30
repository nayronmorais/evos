"""Module contains helpers functions for clustering methods."""

import numpy as np
from matplotlib.patches import Ellipse, Circle
from matplotlib import pyplot as plt
from scipy.stats import chi2


def plot_2d_gauss_cluster(center, inv_cov, gamma=0.99, ls=':', lw=2, 
                          color='green', alpha=0.8, label=None, radii=None, is_inv_cov=True, ax=None):
    """
    Plot a 2D cluster with gaussian shape.

    Parameters
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the cluster.

    inv_cov : numpy.ndarray, shape(n, n)
        Inverse of the covariance matrix.

    gamma : float, optional
        Confidence interval. The default is 0.99.

    ls : str, optional
        The matplotlib line style. Can be any of
        `matplotlib.lines.lineStyles`. The default is `:`.

    lw : float, optional
        The width of the edge (in pt). The default is 2.

    color : str or iterator, optional
        The color of the edge. Can be any matplotlib color name or
        RGB values. The default is `green`.
    
    alpha : float, optional
        Alpha channel for ellipse's contour. The default is 0.8.
        
    label : str, optional
        Label for legends. The default is None.
        
    ax : matplotlib.axes.Axes
        The axes when the ellipse will be draw.

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The cluster representation.

    """
    if inv_cov.shape[0] != 2:
        raise Exception('Available only in 2D problems.')

    if ax is None:
        ax = plt.gca()

    if radii is None:
        radii = chi2.ppf(gamma, 2)

    eigenvalues, eigenvectors = np.linalg.eigh(inv_cov)

    ord_ = np.argsort(eigenvalues)
    ord_ = ord_ if is_inv_cov else ord_[::-1]

    eigenvalues = eigenvalues[ord_]
    eigenvectors = eigenvectors[:, ord_]

    eign = (1 / eigenvalues) if is_inv_cov else eigenvalues
    
    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width = 2 * np.sqrt(radii * eign[0])
    height = 2 * np.sqrt(radii * eign[1])
    

    ellipse = Ellipse(center[0], width, height, angle=theta, fill=False,
                      zorder=10 ** 2, edgecolor=color, ls=ls, lw=lw, alpha=alpha)
    
    ellipse.set_label(label)
    ax.add_artist(ellipse)

    return ellipse


def plot_2d_mfgauss_cluster(c, wh, tau=1e-4, ls=':', lw=2, s=40,
                          color='green', alpha=0.8, label=None, ax=None,):
    """
    Plot a 2D cluster with gaussian shape.

    Parameters
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the cluster.

    inv_cov : numpy.ndarray, shape(n, n)
        Inverse of the covariance matrix.

    gamma : float, optional
        Confidence interval. The default is 0.99.

    ls : str, optional
        The matplotlib line style. Can be any of
        `matplotlib.lines.lineStyles`. The default is `:`.

    lw : float, optional
        The width of the edge (in pt). The default is 2.

    color : str or iterator, optional
        The color of the edge. Can be any matplotlib color name or
        RGB values. The default is `green`.
    
    alpha : float, optional
        Alpha channel for ellipse's contour. The default is 0.8.
        
    label : str, optional
        Label for legends. The default is None.
        
    ax : matplotlib.axes.Axes
        The axes when the ellipse will be draw.

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse
        The cluster representation.

    """
    if c.shape[1] != 2:
        raise Exception('Available only in 2d problems.')

    if ax is None:
        ax = plt.gca()

    
    ax.scatter(*c.T, s=s, marker='P', color=color)
    
    if isinstance(wh, (float, int)) or np.size(wh) == 1:
        ellipse = Circle(c[0], wh, fill=False,
                          zorder=10 ** 2, edgecolor=color, ls=ls, lw=lw, alpha=alpha)
    else:
        ellipse = Ellipse(c[0], wh[0, 0], wh[0, 1], fill=False,
                          zorder=10 ** 2, edgecolor=color, ls=ls, lw=lw, alpha=alpha)
    
    ellipse.set_label(label)
    ax.add_artist(ellipse)

    return ellipse


def plot_3d_gauss_cluster(center, inv_cov, gamma=0.99, ls=':', lw=2, color='m',
                          alpha=0.8, sep_cols=12, sep_rows=12, label=None, ax=None):
    """
    Plot a 3D cluster with gaussian shape.
    
    Parameters
    ----------
    center : numpy.ndarray, shape=(1, n)
        Center of the cluster.

    inv_cov : numpy.ndarray, shape(n, n)
        Inverse of the covariance matrix.

    gamma : float, optional
        Confidence interval. The default is 0.99.

    ls : str, optional
        The matplotlib line style. Can be any of
        `matplotlib.lines.lineStyles`. The default is `:`.

    lw : float, optional
        The width of the edge (in pt). The default is 2.

    color : str or iterator, optional
        The color of the edge. Can be any matplotlib color name or
        RGB values. The default is `green`.
        
    alpha : float, optional
        Alpha channel for ellipse's contour. The default is 0.8.
        
    label : str, optional
        Label for legends. The default is None.
        
    ax : mpl_toolkits.mplot3d.Axes3D
        The axes when the ellipsoid will be draw.

    Returns
    -------
    ellipsoids : mpl_toolkits.mplot3d.art3d.Line3DCollection
        The cluster representation.
        
    Notes
    -----
    Based on approach in https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py.
        
    """
    
    if inv_cov.shape[0] != 3:
        raise Exception('Available only in 3d problems.')
        
    if ax is None:
        ax = plt.gca(projection='3d')
        
    if center.ndim > 1:
        center = center[0, :]

    eigenvalues, eigenvectors = np.linalg.eigh(inv_cov)

    ord_ = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[ord_]
    eigenvectors = eigenvectors[:, ord_]
    
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    radii = chi2.ppf(gamma, 3)
    radii = 2 * np.sqrt(gamma * (1 / eigenvalues))
    
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    
    # Rotation
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j], y[i,j], z[i,j]] = np.dot([x[i,j], y[i,j], z[i,j]], eigenvectors) + center
    
    # plot ellipsoid
    return ax.plot_surface(x, y, z, rstride=sep_cols, cstride=sep_rows, ls='-', lw=1.5, color=color, alpha=alpha)


def plot_decision_region(xmin, xmax, model, nsamples=100, figsize=(10, 10)):
    
    x = np.linspace(xmin, xmax, num=nsamples)
    
    X1, X2 = np.meshgrid(*x.T)
    Z = np.zeros((nsamples, nsamples), dtype=int)
    
    for i in range(nsamples):
        for j in range(nsamples):
            x = np.array([[X1[i, j], X2[i, j]]])
            
            cstar = model.predict(x)
            Z[i, j] = cstar
            
    from matplotlib import pyplot as plt
    from matplotlib import cm
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.contourf(X1, X2, Z, cmap='rainbow', levels=np.unique(Z), zorder=1)
    
    colors = colors = cm.rainbow(np.linspace(0, 1, len(model.clusters)))[::-1]
    for i, (id_, cluster) in enumerate(model.clusters.items()):
        ax.scatter(*cluster.D.T, marker='o', color=colors[i], s=20, edgecolors='k', lw=0.25, zorder=2)
