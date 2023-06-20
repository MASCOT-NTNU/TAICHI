Imagine you have a grid like this:

```
resolution = .01
cdf_z1 = np.arange(-3.5, 3.5, resolution)
cdf_z2 = np.arange(-3.5, 3.5, resolution)
cdf_rho = np.arange(-.999999, 0, .01)
cdf = get_cdf_table(cdf_z1, cdf_z2, cdf_rho)
```

Now you are given a set of grid points:

```
z1 = np.random.rand(-3.5, 3.5, 100)
z2 = np.random.rand(-3.5, 3.5, 100)
rho = np.random.rand(-0.9999, .0, 100)
```

You need to find out the fastest way to interpolate `cdf` for those grid points.

Remember the size of the original grid is big `4GB` in total. So be careful with what method you use.


========================================

Improve the following code in terms of data loading and saving:

```
def get_cdf_table(z1: np.ndarray, z2: np.ndarray, rho: np.ndarray):
    # cdf = np.zeros([len(z1), len(z2), len(rho)])
    # for i in range(z1.shape[0]):
    #     for j in range(z2.shape[0]):
    #         for k in range(rho.shape[0]):
    #             cdf[i, j, k] = multivariate_normal.cdf([z1[i], z2[j]], mean=[0, 0], cov=[[1, rho[k]], [rho[k], 1]])
    cdf = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])
    return cdf

resolution = .1
num_cores = 8
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
pd.DataFrame(dataset, columns=['z1', 'z2', 'rho', 'cdf']).to_csv(os.getcwd()+"/cdf_table.csv", index=False)
```

=======

Here is the code I used to generate the cdf table:

```
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

resolution = .01
num_cores = 48
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
df = pd.DataFrame(dataset, columns=['z1', 'z2', 'rho', 'cdf'])
t1 = time()
df.to_feather(os.getcwd()+"/cdf_table.feather")
t2 = time()
print(f"Time elapsed: {t2-t1:.2f} seconds.")
```

I am using python 3.8.10, then I got an error: TypeError: Argument 'table' has incorrect type (expected pyarrow.lib.Table, got DataFrame)

How to fix it?


============

I have two functions, they should output the same result, but now when the grid size is bigger, then the results are different. And I do not understand why?

```
def __get_eibv_analytical(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray,
                          threshold: float) -> float:
    """
    Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

    Input:
        - mu: mean of the posterior marginal distribution.
            - np.array([mu1, mu2, ...])
        - sigma_diag: diagonal elements of the posterior marginal variance.
            - np.array([sigma1, sigma2, ...])
        - vr_diag: diagonal elements of the variance reduction.
            - np.array([vr1, vr2, ...])
    """
    eibv = .0
    EIBV = []
    print("Analyzing...")
    for i in range(len(mu)):
        sn2 = sigma_diag[i]
        vn2 = vr_diag[i]

        sn = np.sqrt(sn2)
        m = mu[i]

        mur = (threshold - m) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        eibv_temp = multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                            np.array([[sig2r_1, -sig2r], [-sig2r, sig2r_1]]).squeeze())
        EIBV.append(eibv_temp)
        eibv += eibv_temp
    return eibv, EIBV

@staticmethod
def __get_eibv_analytical_fast(mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray,
                               threshold: float) -> float:
    """
    Calculate the eibv using the analytical formula but using a loaded cdf dataset.
    """
    print("Fasting...")
    t1 = time.time()

    sn2 = sigma_diag
    vn2 = vr_diag
    sn = np.sqrt(sn2)
    mur = (np.ones_like(mu) * threshold - mu) / sn
    sig2r_1 = sn2 + vn2
    sig2r = vn2
    z1 = mur
    z2 = -mur
    rho = -sig2r / sig2r_1
    grid = np.stack((z1, z2, rho), axis=1)

    t2 = time.time()
    print("Constructing grid takes: ", t2 - t1, "s")
    start_time = time.time()
    distances, indices = GMRF.__CDF_TABLE_TREE.query(grid)
    print("distance matrix ime consumed: ", time.time() - start_time, "s")
    eibv_temp = GMRF.__CDF_TABLE[indices, 3]
    eibv = np.sum(eibv_temp)
    return eibv, eibv_temp
```

The __CDF_TABLE_TREE or __CDF_TABLE are precomputed using multivariate_normal.cdf function for the following range of z1, z2 and rho:

```
cdf_z1 = np.arange(-3.5, 3.5, resolution)
cdf_z2 = np.arange(-3.5, 3.5, resolution)
cdf_rho = np.arange(-.999999, 0, .1)
```

So I don't understand why they output the different result.

==========

I need to build a dataset that contains probability p for given (rho, z1, z2).

I want to precompute values for the following z1, z2 and rho: rho = np.arange(-.9999, 0, .01) and z1=np.arange(-3.5, 3.5, .01), and z2 = np.arange(-3.5, 3.5, .01).

Decide on which data structure and file format to save the data and load the data.

Later I need to query p given (rho, z1, z2). It should choose the closest rho, and then use z1 and z2 to find the corresponding point. (rho, z1, z2) can be any real numbers, so do not expect you will get the exact same rho as in the precomputed ones.

Choose the best data structure and best library.

=========

You have the following function, and you have to make them faster using multiple cores such as joblib, and Parallel and delayed or other libraries if you think they are better:
```
import numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator
import joblib

# Function to calculate the CDF value
def get_cdf_value(z1, z2, rho):
    return multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])

# Parameters
resolution = .1
rho_values = np.arange(-.9999, 0, resolution)
z1_values = np.arange(-3.5, 3.5, resolution)
z2_values = np.arange(-3.5, 3.5, resolution)

# Preallocate an array for the results
cdf_values = np.empty((rho_values.size, z1_values.size, z2_values.size))

# Compute the CDF values for each combination of rho, z1, z2
for i, rho in enumerate(rho_values):
    for j, z1 in enumerate(z1_values):
        for k, z2 in enumerate(z2_values):
            cdf_values[i, j, k] = get_cdf_value(z1, z2, rho)

# Create an interpolator for each rho layer
interpolators = [RegularGridInterpolator((z1_values, z2_values), cdf_values[i, :, :], bounds_error=False, fill_value=None) for i in range(rho_values.size)]

joblib.dump(interpolators, "interpolator.joblib")
```
