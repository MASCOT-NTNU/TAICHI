"""
This script calculates the CDF values for the bivariate normal distribution.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-25

Methodology:
    1. Calculate the CDF values for each combination of rho, z1, z2
    2. Calculate the distance between the query point and each grid point
    3. Find the closest grid point and interpolate the value

"""
import numpy as np
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
import os
from time import time
import pandas as pd

def get_cdf_table(z1: float, z2: float, rho: float):
    cdf = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return cdf

resolution = .1
num_cores = 1
cdf_z1 = np.arange(-3.5, 3.5, resolution)
cdf_z2 = np.arange(-3.5, 3.5, resolution)
cdf_rho = np.arange(-.999999, 0, .1)

t1 = time()
res = Parallel(n_jobs=num_cores)(delayed(get_cdf_table)(z1=z1, z2=z2, rho=rho) for z1 in cdf_z1 for z2 in cdf_z2 for rho in cdf_rho)
t2 = time()
print(f"Data size: {len(res)}, Time elapsed: {t2-t1:.2f} seconds.")

# cdf = get_cdf_table(cdf_z1, cdf_z2, cdf_rho)
grid = []
for z1 in cdf_z1:
    for z2 in cdf_z2:
        for rho in cdf_rho:
            grid.append([z1, z2, rho])
grid = np.array(grid)
dataset = np.hstack((grid, np.array(res).reshape(-1, 1)))
# np.savez(os.getcwd()+"/cdf_table.npz", dataset)
df = pd.DataFrame(dataset, columns=['z1', 'z2', 'rho', 'cdf'])
t1 = time()
df.to_feather(os.getcwd()+"/cdf_table.feather")
t2 = time()
print(f"Time elapsed: {t2-t1:.2f} seconds.")
