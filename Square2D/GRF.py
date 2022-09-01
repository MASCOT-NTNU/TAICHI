"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from usr_func import interpolate_2d, vectorise
from scipy.spatial.distance import cdist
from TAICHI.Square2D.Config.Config import FILEPATH, NUGGET, LATERAL_RANGE, SIGMA, CMAP


class GRF:

    def __init__(self, seed=None):
        np.random.seed(seed)
        self.load_grf_grid()
        self.load_prior_mean()
        self.get_covariance_matrix()
        self.get_simulated_truth()
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        print("GRF1-4 is set up successfully!")

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        self.N = self.grf_grid.shape[0]
        print("GRF1: Grid is loaded successfully!")

    def load_prior_mean(self):
        self.mu_prior = pd.read_csv(FILEPATH + "Config/mu_prior.csv")['value'].to_numpy().reshape(-1, 1)
        print("GRF2: Prior mean is loaded successfully!")

    def get_covariance_matrix(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = np.sqrt(NUGGET)
        distance_matrix = cdist(self.grf_grid, self.grf_grid)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * distance_matrix) * np.exp(-self.eta * distance_matrix)
        print("GRF3: Covariance matrix is computed successfully!")

    def get_simulated_truth(self):
        self.mu_truth = (self.mu_prior.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior)).reshape(-1, 1))
        print("GRF4: Simulated truth is computed successfully!")

    def update_grf_model(self,  ind_measured=np.array([1, 2]), salinity_measured=vectorise([0, 0])):
        m = len(ind_measured)
        t1 = time.time()
        F = np.zeros([m, self.N])
        for i in range(m):
            F[i, ind_measured[i]] = True
        R = np.eye(m) * self.tau ** 2
        C = F @ self.Sigma_cond @ F.T + R
        self.mu_cond = self.mu_cond + self.Sigma_cond @ F.T @ np.linalg.solve(C, (salinity_measured - F @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ F.T @ np.linalg.solve(C, F @ self.Sigma_cond)
        t2 = time.time()
        print("GRF model updates takes: ", t2 - t1)

    def get_ind_from_location(self, x, y):
        dx = (self.grf_grid[:, 0] - x)**2
        dy = (self.grf_grid[:, 1] - y)**2
        dd = dx + dy
        ind = np.argmin(dd)
        return ind

    def get_posterior_variance_at_ind(self, ind):
        self.SF = self.Sigma_cond[:, ind].reshape(-1, 1)
        self.MD = 1 / (self.Sigma_cond[ind, ind] + NUGGET)
        self.VR = self.SF @ self.SF.T * self.MD
        self.SP = self.Sigma_cond - self.VR
        return np.diag(self.SP)

    def check_prior(self):
        # plt.scatter(self.grf_grid[:, 0], self.grf_grid[:, 1], c=self.mu_prior,
        #             cmap=get_cmap("BrBG", 10), s=150)
        # plt.colorbar()
        # plotf_vector(self.grf_grid[:, 0], self.grf_grid[:, 1], values=self.mu_prior)
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_prior, cmap=get_cmap("BrBG", 10))
        plt.show()

    def check_update(self):
        plt.imshow(self.Sigma_cond)
        plt.colorbar()
        plt.show()
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_prior)
        plt.show()
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_truth)
        plt.show()

        N = 10
        ind = np.random.randint(0, self.grf_grid.shape[0], N)
        val = np.random.uniform(0, 1, N)
        self.update_grf_model(ind, vectorise(val))
        x = self.grf_grid[:, 0]
        y = self.grf_grid[:, 1]
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_cond)
        plt.plot(x[ind], y[ind], 'k.')
        plt.subplot(122)
        plt.scatter(x, y, c=np.diag(self.Sigma_cond), cmap=CMAP)
        plt.colorbar()
        # self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=np.diag(self.Sigma_cond), vmin=0, vmax=.01)
        plt.plot(x[ind], y[ind], 'k.')
        plt.show()

        N = 20
        ind = np.random.randint(0, self.grf_grid.shape[0], N)
        val = np.random.uniform(0, 1, N)
        self.update_grf_model(ind, vectorise(val))
        x = self.grf_grid[:, 0]
        y = self.grf_grid[:, 1]
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_cond)
        plt.plot(x[ind], y[ind], 'k.')
        plt.subplot(122)
        # self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=np.diag(self.Sigma_cond), vmin=0, vmax=.01)
        plt.scatter(x, y, c=np.diag(self.Sigma_cond), cmap=CMAP)
        plt.colorbar()
        plt.plot(x[ind], y[ind], 'k.')
        plt.show()

        N = 30
        ind = np.random.randint(0, self.grf_grid.shape[0], N)
        val = np.random.uniform(0, 1, N)
        self.update_grf_model(ind, vectorise(val))
        x = self.grf_grid[:, 0]
        y = self.grf_grid[:, 1]
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_cond)
        plt.plot(x[ind], y[ind], 'k.')
        plt.subplot(122)
        # self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=np.diag(self.Sigma_cond), vmin=0, vmax=.01)
        plt.scatter(x, y, c=np.diag(self.Sigma_cond), cmap=CMAP)
        plt.colorbar()
        plt.plot(x[ind], y[ind], 'k.')
        plt.show()

        # self.update_grf_model(10, 30)
        # plt.figure(figsize=(20, 10))
        # plt.subplot(121)
        # self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=self.mu_cond)
        # plt.subplot(122)
        # self.plotf(self.grf_grid[:, 0], self.grf_grid[:, 1], value=np.diag(self.Sigma_cond), vmin=0, vmax=.01)
        # plt.show()

    def plotf(self, x, y, value, cmap=CMAP, vmin=None, vmax=None):
        gx, gy, v = interpolate_2d(x, y, 100, 100, value, "cubic")
        plt.scatter(gx, gy, c=v, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        pass



if __name__ == "__main__":
    grf = GRF()
    # grf.check_prior()
    grf.check_update()


