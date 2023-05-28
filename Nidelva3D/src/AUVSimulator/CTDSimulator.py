"""
CTDSimulator simulates the CTD sensor sampling in the ground truth field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28

"""
from GRF.GRF import GRF
import numpy as np
from typing import Union
from pykdtree.kdtree import KDTree


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    def __init__(self, random_seed: int = 0):
        """
        Set up the CTD simulated truth field.
        """
        np.random.seed(random_seed)

        grf = GRF()
        self.grid = grf.get_grid()
        mu_prior = grf.get_mu()
        Sigma_prior = grf.get_covariance_matrix()
        self.mu_truth = mu_prior + np.linalg.cholesky(Sigma_prior) @ np.random.randn(len(mu_prior)).reshape(-1, 1)
        self.__field = np.hstack((self.grid, self.mu_truth))
        self.__field_grid = self.grid
        self.__field_grid_tree = KDTree(self.__field_grid)
        self.__field_salinity = self.__field[:, -1]

    def __get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                return self.__field_grid_tree.query(loc.reshape(1, -1))[1]
            elif dm == 2:
                return self.__field_grid_tree.query(loc)[1]
            else:
                return None
        else:
            return None

    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y, z]])

        Returns:
            salinity value at loc
        """
        ind = self.__get_ind_from_location(loc)
        if ind is not None:
            return self.__field_salinity[ind]
        else:
            return None


if __name__ == "__main__":
    c = CTDSimulator()


