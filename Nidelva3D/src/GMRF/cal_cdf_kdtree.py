"""
This script is used to calculate the CDF values for the bivariate normal distribution.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-25

Methodology:
    1. Calculate the CDF values for each combination of rho, z1, z2
    2. Create an interpolator for each rho layer using KDTree
    3. Later, when you have many query points, find the closest rho layer and interpolate the value

"""
import os
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import KDTree
import xarray as xr
from joblib import Parallel, delayed

# Function to calculate the CDF value
def get_cdf_value(z1, z2, rho):
    return multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])

# Function to compute the CDF values for a given rho
def compute_cdf_for_rho(rho, z1_values, z2_values):
    cdf_values_rho = np.empty((z1_values.size, z2_values.size))
    for j, z1 in enumerate(z1_values):
        for k, z2 in enumerate(z2_values):
            cdf_values_rho[j, k] = get_cdf_value(z1, z2, rho)
    return cdf_values_rho

# Parameters
resolution = .1
rho_values = np.arange(-.9999, 0, resolution)
z1_values = np.arange(-3.5, 3.5, resolution)
z2_values = np.arange(-3.5, 3.5, resolution)

# Use joblib to run the computation in parallel
n_jobs = -1  # Use all available cores
cdf_values = Parallel(n_jobs=n_jobs)(delayed(compute_cdf_for_rho)(rho, z1_values, z2_values) for rho in rho_values)

# Convert the list of arrays to a 3D array
cdf_values = np.array(cdf_values)

# Create the xarray Dataset
ds = xr.Dataset(
    {'p': (('rho', 'z1', 'z2'), cdf_values)},
    coords={'rho': rho_values, 'z1': z1_values, 'z2': z2_values}
)

# Save the dataset to a NetCDF file
ds.to_netcdf('cdf_table.nc')

# Clean up
del ds, cdf_values


#%%
# --- Later, when you need to query the data ---

import os
import numpy as np
import xarray as xr
from sklearn.neighbors import KDTree

resolution = .01
rho_values = np.arange(-.9999, 0, resolution)
z1_values = np.arange(-3.5, 3.5, resolution)
z2_values = np.arange(-3.5, 3.5, resolution)

# Load the data from the NetCDF file
ds = xr.open_dataset(os.getcwd() + '/GMRF/cdf_table.nc')

# Build a KDTree for each layer of rho
# Build a KDTree for each layer of rho
kdtrees = {}
for rho in ds['rho']:
    layer = ds.sel(rho=rho.item())
    # Prepare the data for the KDTree
    X = np.stack(np.meshgrid(layer['z1'], layer['z2']), -1).reshape(-1, 2)
    # Create the KDTree and store it in the dictionary
    kdtrees[rho.item()] = KDTree(X)

# kdtrees = {}
# for rho in ds['rho']:
#     layer = ds.sel(rho=rho.item())
#     # Prepare the data for the KDTree
#     X = np.stack(np.meshgrid(layer['z1'], layer['z2']), -1).reshape(-1, 2)
#     # Create the KDTree and store it in the dictionary
#     kdtrees[rho.item()] = KDTree(X)

# Function to query the data for a given rho, z1, z2
# Function to query the data for a given rho, z1, z2
#%%
def query(rho, z1, z2):
    # Find the closest rho layer
    closest_rho = ds['rho'].sel(rho=rho, method='nearest').item()
    # Use the KDTree for this layer to find the closest point
    dist, ind = kdtrees[closest_rho].query(np.array([[z1, z2]]))
    # Convert the 1D index to 2D indices
    z1_ind, z2_ind = np.unravel_index(ind, (z1_values.size, z2_values.size))
    # Retrieve the corresponding probability
    p = ds['p'].sel(rho=closest_rho, z1=z1_values[z1_ind[0]], z2=z2_values[z2_ind[0]]).item()
    return p

# p = query(-0.5, 1.0, 1.0)
# print(p)

# import pandas as pd
# import numpy as np
# import os
# cdf = pd.read_feather(os.getcwd() + "/GMRF/cdf_table.feather").to_numpy()
# rho = np.unique(cdf[:, 2])
# z1 = np.unique(cdf[:, 0])
# z2 = np.unique(cdf[:, 1])

#%%
# import numpy as np
# from scipy.stats import multivariate_normal
# from joblib import Parallel, delayed
# import os
# from time import time
# import pandas as pd
#
# def get_cdf_table(z1: float, z2: float, rho: float):
#     cdf = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])
#     return cdf
#
# resolution = .1
# num_cores = 1
# cdf_z1 = np.arange(-3.5, 3.5, resolution)
# cdf_z2 = np.arange(-3.5, 3.5, resolution)
# cdf_rho = np.arange(-.999999, 0, .1)
#
# t1 = time()
# res = Parallel(n_jobs=num_cores)(delayed(get_cdf_table)(z1=z1, z2=z2, rho=rho) for z1 in cdf_z1 for z2 in cdf_z2 for rho in cdf_rho)
# t2 = time()
# print(f"Data size: {len(res)}, Time elapsed: {t2-t1:.2f} seconds.")
#
# # cdf = get_cdf_table(cdf_z1, cdf_z2, cdf_rho)
# grid = []
# for z1 in cdf_z1:
#     for z2 in cdf_z2:
#         for rho in cdf_rho:
#             grid.append([z1, z2, rho])
# grid = np.array(grid)
# dataset = np.hstack((grid, np.array(res).reshape(-1, 1)))
# # np.savez(os.getcwd()+"/cdf_table.npz", dataset)
# df = pd.DataFrame(dataset, columns=['z1', 'z2', 'rho', 'cdf'])
# t1 = time()
# df.to_feather(os.getcwd()+"/cdf_table.feather")
# t2 = time()
# print(f"Time elapsed: {t2-t1:.2f} seconds.")
