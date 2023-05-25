"""
This script calculates the CDF values for the bivariate normal distribution.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-25

Methodology:
    1. Calculate the CDF values for each combination of rho, z1, z2
    2. Create an interpolator for each rho layer using Regular Grid Interpolator
    3. Later, when you have many query points, find the closest rho layer and interpolate the value
"""
import os

import numpy as np
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
import joblib

# Function to calculate the CDF value
def get_cdf_value(z1, z2, rho):
    return multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=[[1, rho], [rho, 1]])

# Parameters
resolution = .1
rho_values = np.arange(-.9999, 0, resolution)
z1_values = np.arange(-3.5, 3.5, resolution)
z2_values = np.arange(-3.5, 3.5, resolution)

# Function to compute the CDF values for one rho
def compute_one_rho(rho, z1_values, z2_values):
    cdf_values_one_rho = np.empty((z1_values.size, z2_values.size))
    for i, z1 in enumerate(z1_values):
        for j, z2 in enumerate(z2_values):
            cdf_values_one_rho[i, j] = get_cdf_value(z1, z2, rho)
    return cdf_values_one_rho

# Compute the CDF values for each rho in parallel
results = Parallel(n_jobs=-1)(delayed(compute_one_rho)(rho, z1_values, z2_values) for rho in rho_values)

# Build the full cdf_values array
cdf_values = np.stack(results)

joblib.dump((rho_values, z1_values, z2_values, cdf_values), "interpolator_medium.joblib")

#%%
from scipy.interpolate import RegularGridInterpolator
import joblib
import os
import numpy as np


rho_values, z1_values, z2_values, cdf_values = joblib.load(os.getcwd() + "/GMRF/interpolator_medium.joblib")

interpolators = [RegularGridInterpolator((z1_values, z2_values), cdf_values[i, :, :], bounds_error=False, fill_value=None) for i in range(rho_values.size)]


#%%
# Later, when you have many query points:
z1 = np.arange(-3.5, 3.5, .1)
z2 = np.arange(-3.5, 3.5, .1)
rho = np.ones_like(z1) * .0
grid = []
for i in range(len(rho)):
    for j in range(len(z1)):
        for k in range(len(z2)):
            grid.append([rho[i], z1[j], z2[k]])
query_points = np.array(grid)

# Function to find the closest rho layer and interpolate the value
def query(rho, z1, z2):
    # Find the index of the closest rho layer
    i = np.abs(rho_values - rho).argmin()
    # Use the interpolator for this layer to interpolate the value
    return interpolators[i]([[z1, z2]])

p_values = np.array([query(rho, z1, z2) for rho, z1, z2 in query_points])

#%%
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
plt.scatter(query_points[:, 1], query_points[:, 2], c=p_values[:, 0], cmap=get_cmap("RdBu", 10), vmin=0, vmax=1)
plt.colorbar()
plt.show()

