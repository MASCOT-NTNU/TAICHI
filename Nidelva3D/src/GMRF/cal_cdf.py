import numpy as np
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
import os
from time import time
import pandas as pd

def get_cdf_table(z1: np.ndarray, z2: np.ndarray, rho: np.ndarray):
    # cdf = np.zeros([len(z1), len(z2), len(rho)])
    # for i in range(z1.shape[0]):
    #     for j in range(z2.shape[0]):
    #         for k in range(rho.shape[0]):
    #             cdf[i, j, k] = multivariate_normal.cdf([z1[i], z2[j]], mean=[0, 0], cov=[[1, rho[k]], [rho[k], 1]])
    cdf = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return cdf

resolution = .5
num_cores = 1
cdf_z1 = np.arange(-3.5, 3.5, resolution)
cdf_z2 = np.arange(-3.5, 3.5, resolution)
cdf_rho = np.arange(-.999999, 0, .01)

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
pd.DataFrame(dataset, columns=['z1', 'z2', 'rho', 'cdf']).to_csv(os.getcwd()+"/cdf_table.csv", index=False)
