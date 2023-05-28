"""
CTD module simulates the CTD sensor sampling in the ground truth field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28

Methodology:
    1. Generate a Gaussian Random Field (GRF) with a given covariance matrix.
    2. Precalculate the path from the starting location to the ending location.
    3. Sample the GRF at each location along the path.
"""
from GRF.GRF import GRF
from pykdtree.kdtree import KDTree
import numpy as np
from typing import Union


class CTD:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    def __init__(self, loc_start: np.ndarray = np.array([0, 0, 0]), random_seed: int = 0):

        """
        Set up the CTD simulated truth field.
        TODO: Check the starting location, it can induce serious problems.
        """
        np.random.seed(random_seed)

        self.grf = GRF()
        self.grid = self.grf.get_grid()
        self.grid_tree = KDTree(self.grid)
        mu_prior = self.grf.get_mu()
        Sigma_prior = self.grf.get_covariance_matrix()
        self.mu_truth = mu_prior + np.linalg.cholesky(Sigma_prior) @ np.random.randn(len(mu_prior)).reshape(-1, 1)

        """
        Set up CTD data gathering
        """
        self.loc_now = loc_start
        self.loc_prev = np.array([0, 0, 0])
        self.ctd_data = np.empty([0, 4])  # np.array([x, y, z, sal])
        self.speed = 1.5  # m/s

    def get_ctd_data_1hz(self, loc: np.ndarray) -> np.ndarray:
        """
        Simulate CTD data gathering at 1hz.
        It means that it returns only one sample.
        Args:
            loc: np.array([x, y, z])
        Return:
            data: np.array([x, y, z, sal])
        """
        sal = self.get_salinity_at_loc(loc.reshape(1, -1))
        self.ctd_data = np.stack((loc[0], loc[1], loc[2], sal.flatten()[0])).reshape(1, -1)
        return self.ctd_data

    def get_ctd_data(self, loc: np.ndarray) -> np.ndarray:
        """
        Simulate CTD data gathering.
        Args:
            loc: np.array([x, y, z])
        Return:
            dataset: np.array([x, y, z, sal])
        """
        self.loc_prev = self.loc_now
        self.loc_now = loc
        x_start, y_start, z_start = self.loc_prev
        x_end, y_end, z_end = self.loc_now
        N = 20
        if N != 0:
            x_path = np.linspace(x_start, x_end, N)
            y_path = np.linspace(y_start, y_end, N)
            z_path = np.linspace(z_start, z_end, N)
            loc = np.stack((x_path, y_path, z_path), axis=1)
            sal = self.get_salinity_at_loc(loc)
            self.ctd_data = np.stack((x_path, y_path, sal.flatten()), axis=1)
        return self.ctd_data

    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y, z]])

        Returns:
            salinity value at loc
        """
        dist, ind = self.grid_tree.query(loc)
        if ind is not None:
            return self.mu_truth[ind]
        else:
            return None

    def get_ground_truth(self) -> np.ndarray:
        """ Return ground truth. """
        return self.mu_truth