import numpy as np


def toy_dataset(num_samples=1000, noise=None, seed=None):
    
    rd_gen = np.random.default_rng(0 if seed is None else seed)
    
    n_c1 = int(num_samples / 2)
    n_c2 = num_samples - n_c1
    
    x1 = rd_gen.uniform(-10, 10, size=n_c1) + rd_gen.normal(loc=0.0, scale=1.0, size=n_c1)
    x2 = rd_gen.uniform(-8, 8, size=n_c1) * rd_gen.normal(scale=0.15, size=n_c1) + np.cos(x1 / np.pi)
    
    Xc1 = np.c_[x1, x2]
    
    Xc2_1 = rd_gen.normal(loc=[-10, 10], scale=0.1, size=(n_c2, 2))
    Xc2_2 = rd_gen.normal(loc=[15, 5], scale=0.001, size=(n_c2, 2))
    Xc2_3 = rd_gen.normal(loc=[16, 12.5], scale=0.75, size=(n_c2, 2))
    w = rd_gen.uniform(size=(n_c2, 3))
    w /= w.sum(axis=1, keepdims=True)
    
    Xc2 = w[:, [0]]*Xc2_1 + w[:, [1]]*Xc2_2 +  w[:, [2]]*Xc2_3
    
    if noise is not None and noise > 0:
        noise_c1 = rd_gen.normal(loc=0.0, scale=noise, size=(n_c1, 2))
        noise_c2 = rd_gen.normal(loc=0.0, scale=noise, size=(n_c2, 2))
        
        Xc1 += noise_c1
        Xc2 += noise_c2
        
    X = np.r_[Xc1, Xc2]
    Y = np.zeros(shape=num_samples, dtype=np.uint8)
    Y[n_c2:] = 1
    
    return X, Y
