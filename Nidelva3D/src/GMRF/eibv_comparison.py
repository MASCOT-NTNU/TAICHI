"""
This script compares different methods for computing the expected integrated Bernoulli variance (EIBV)
of a Gaussian Markov Random Field (GMRF).

Methodology:
    1. Generate a GRF with a given covariance matrix.
    2. Sample the GRF at a given number of locations.
"""
import os
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.cm import get_cmap
from time import time
from numba import njit


"""
Set up the GRF field. 
"""
sigma = 1.0
nugget = .4
eta = 4.5 / .7
threshold = 27.8

N = 25
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

grid = np.stack((X.flatten(), Y.flatten()), axis=1)
Ngrid = grid.shape[0]

dm = cdist(grid, grid, metric='euclidean')
Sigma = sigma ** 2 * (1 + eta * dm) * np.exp(-eta * dm)

mu = np.linalg.cholesky(Sigma) @ np.random.randn(Sigma.shape[0]).reshape(-1, 1)
mu += np.ones_like(mu) * 28

plt.scatter(grid[:, 0], grid[:, 1], c=mu, cmap=get_cmap("BrBG", 10))
plt.colorbar()
plt.show()

#%%
def get_cdf_table(z1: np.ndarray, z2: np.ndarray, rho: np.ndarray):
    cdf = np.zeros([len(z1), len(z2), len(rho)])
    for i in range(z1.shape[0]):
        for j in range(z2.shape[0]):
            for k in range(rho.shape[0]):
                cdf[i, j, k] = multivariate_normal.cdf([z1[i], z2[j]], mean=[0, 0], cov=[[1, rho[k]], [rho[k], 1]])
    return cdf

# cdf_z1 = np.arange(-3.5, 3.5, .05)
# cdf_z2 = np.arange(-3.5, 3.5, .05)
# cdf_rho = np.arange(-.999999, 0, .01)
# cdf = get_cdf_table(cdf_z1, cdf_z2, cdf_rho)
# np.savez(os.getcwd()+"/cdf_table.npz", z1=cdf_z1, z2=cdf_z2, rho=cdf_rho, cdf=cdf)

df_table = np.load(os.getcwd() + "/cdf_table.npz")
cdf_z1 = df_table['z1']
cdf_z2 = df_table['z2']
cdf_rho = df_table['rho']
cdf = df_table['cdf']

#%%
def get_eibv_analytical(mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
    """
    Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.
    """
    eibv = .0
    for i in range(len(mu)):
        sn2 = sigma_diag[i]
        vn2 = vr_diag[i]

        sn = np.sqrt(sn2)
        m = mu[i]

        mur = (threshold - m) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        eibv += multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                        np.array([[sig2r_1, -sig2r],
                                                  [-sig2r, sig2r_1]]).squeeze())
    return eibv

@njit
def get_eibv_analytical_fast(mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray,
                             threshold: float, cdf_z1: np.ndarray, cdf_z2: np.ndarray,
                             cdf_rho: np.ndarray, cdf_table: np.ndarray) -> float:
    """
    Calculate the eibv using the analytical formula but using a loaded cdf dataset.
    """
    eibv = .0
    for i in range(len(mu)):
        sn2 = sigma_diag[i]
        vn2 = vr_diag[i]

        sn = np.sqrt(sn2)
        m = mu[i]

        mur = (threshold - m) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1

        ind1 = np.argmin(np.abs(z1 - cdf_z1))
        ind2 = np.argmin(np.abs(z2 - cdf_z2))
        ind3 = np.argmin(np.abs(rho - cdf_rho))

        eibv += cdf_table[ind1][ind2][ind3]
        # eibv += multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
        #                                 np.array([[sig2r_1, -sig2r],
        #                                           [-sig2r, sig2r_1]]).squeeze())
    return eibv

sigma_diag = np.diag(Sigma)
vr_diag = sigma_diag
get_eibv_analytical_fast(mu, sigma_diag, vr_diag, threshold, cdf_z1, cdf_z2, cdf_rho, cdf)

#%%
# ind_measureds = np.random.choice(Ngrid, 5, replace=False)
ind_measureds = [10]
t1 = []
t2 = []
eibv_1 = []
eibv_2 = []

for ind_measured in ind_measureds:
    F = np.zeros([1, Ngrid])
    F[0, ind_measured] = True
    R = np.eye(1) * nugget
    C = F @ Sigma @ F.T + R
    VR = Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
    Sigma_posterior = Sigma - VR

    sigma_diag = np.diag(Sigma_posterior)
    vr_diag = np.diag(VR)
    mu_input = mu.squeeze()

    tic = time()
    eibv1 = get_eibv_analytical(mu_input, sigma_diag, vr_diag)
    toc = time()
    t1.append(toc - tic)
    eibv_1.append(eibv1)

    tic = time()
    eibv2 = get_eibv_analytical_fast(mu_input, sigma_diag, vr_diag, threshold, cdf_z1, cdf_z2, cdf_rho, cdf)
    toc = time()
    t2.append(toc - tic)
    eibv_2.append(eibv2)

print("Average time for analytical: ", np.mean(t1))
print("Average time for analytical fast: ", np.mean(t2))
# print("Total difference: ", np.sum(diff))

#%%
plt.figure(figsize=(5, 5))
plt.plot(eibv_1, eibv_2, '.')
plt.plot([43, 52], [43, 52], 'r--')
plt.axis('equal')
plt.show()
