"""
Gaussian Random Field module handles the data assimilation and the cost valley computation.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

Objectives:
    1. Construct the Gaussian Random Field (GRF) kernel.
    2. Update the prior mean and covariance matrix.
    3. Assimilate in-situ data.

Methodology:
    1. Construct the GRF kernel.
        1.1. Construct the distance matrix using
            .. math::
                d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + \ksi^2 (z_i - z_j)^2}
        1.2. Construct the covariance matrix.
            .. math::
                \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})
    2. Update the prior mean and covariance matrix.
        2.1. Update the prior mean.
        2.2. Update the prior covariance matrix.
    3. Calculate the EIBV for given locations.
        3.1. Compute the EIBV for given locations.

"""
from GMRF.spde import spde
from WaypointGraph import WaypointGraph
from WGS import WGS
from SINMOD import SINMOD
from usr_func.checkfolder import checkfolder
from usr_func.sort_polygon_vertices import sort_polygon_vertices
import numpy as np
from pykdtree.kdtree import KDTree
from typing import Union
from scipy.stats import norm, multivariate_normal
from joblib import Parallel, delayed
import time
import pandas as pd
import os


class GRF:
    """
    Gaussian Random Field module handles the data assimilation and EIBV calculation.
    """
    __MIN_DEPTH_FOR_DATA_ASSIMILATION = .25
    __GRF_NEIGHBOUR_DISTANCE = 32
    __LATERAL_RANGE = 550
    __VERTICAL_RANGE = 2
    __KSI = __LATERAL_RANGE / __VERTICAL_RANGE
    __GRF_DEPTHS = np.array([.5, 1.5, 2.5, 3.5, 4.5, 5.5])

    def __init__(self) -> None:
        """
        Set up the Gaussian Random Field (GRF) kernel.
        """
        self.__create_data_folders()
        self.__cnt = 0

        self.__spde = spde()
        self.__sinmod = SINMOD()

        """ Empirical parameters """
        self.__sigma = 1.
        self.__nugget = .4
        self.__threshold = self.__spde.threshold
        self.__eta = 4.5 / self.__LATERAL_RANGE  # decay factor
        self.__tau = np.sqrt(self.__nugget)  # measurement noise

        self.__construct_grf_kernel()
        self.__construct_prior_mean()
        self.__mu_prior = self.__mu
        self.__Sigma_prior = self.__Sigma

    def __create_data_folders(self) -> None:
        """
        Create data folders.

        Method:
            1. Create a folder for the assimilated data.
            2. Create a folder for the CTD data.
            3. Create a folder for the threshold data.
        """
        t = int(time.time())
        f = os.getcwd()
        self.__foldername = f + "/GRF/data/{:d}/".format(t)
        self.__foldername_ctd = f + "/GRF/raw_ctd/{:d}/".format(t)
        # self.__foldername_thres = f + "/GRF/threshold/{:d}/".format(t)
        checkfolder(self.__foldername)
        checkfolder(self.__foldername_ctd)
        # checkfolder(self.__foldername_thres)

    def __construct_grf_kernel(self) -> None:
        """
        Construct distance matrix and thus Covariance matrix for the kernel.

        Methodology:
            1. Construct the distance matrix using
                .. math::
                    d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + \ksi^2 (z_i - z_j)^2}
            2. Construct the covariance matrix.
                .. math::
                    \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})

        """
        # s1, get the grid for the GRF kernel. The grid is a 3D array with shape (n, 3).
        polygon_border_wgs = np.load(os.getcwd() + "/GMRF/models/grid.npy")[:, 2:]
        polygon_border = np.stack((WGS.latlon2xy(polygon_border_wgs[:, 0], polygon_border_wgs[:, 1])), axis=1)
        polygon_border = sort_polygon_vertices(polygon_border)
        polygon_obstacles = [[[]]]
        waypoint_graph = WaypointGraph(neighbour_distance=self.__GRF_NEIGHBOUR_DISTANCE,
                                       depths=self.__GRF_DEPTHS,
                                       polygon_border=polygon_border,
                                       polygon_obstacles=polygon_obstacles)
        self.__grf_grid = waypoint_graph.get_waypoints()
        self.__grf_grid_kdtree = KDTree(self.__grf_grid)
        self.__n_grf_grid = self.__grf_grid.shape[0]

        # s2, construct the distance matrix.
        def __cal_distance_matrix(grid1, grid2, ksi):
            dx = (grid1[:, 0].reshape(-1, 1) @ np.ones((1, grid2.shape[0])) -
                  np.ones((grid1.shape[0], 1)) @ grid2[:, 0].reshape(1, -1))
            dy = (grid1[:, 1].reshape(-1, 1) @ np.ones((1, grid2.shape[0])) -
                  np.ones((grid1.shape[0], 1)) @ grid2[:, 1].reshape(1, -1))
            dz = (grid1[:, 2].reshape(-1, 1) @ np.ones((1, grid2.shape[0])) -
                  np.ones((grid1.shape[0], 1)) @ grid2[:, 2].reshape(1, -1))
            d = np.sqrt(dx ** 2 + dy ** 2 + (ksi * dz) ** 2)
            return d
        self.__distance_matrix = __cal_distance_matrix(self.__grf_grid, self.__grf_grid, self.__KSI)

        # s3, construct the covariance matrix.
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def __construct_prior_mean(self) -> None:
        """
        Construct prior mean for the kernel.

        Methodology:
            1. Construct the prior mean using the SINMOD dataset.
            2. Interpolate the prior mean onto the grid.

        Returns:
            None
        """
        lat, lon = WGS.xy2latlon(self.__grf_grid[:, 0], self.__grf_grid[:, 1])
        coordinates = np.stack((lat, lon, self.__grf_grid[:, 2]), axis=1)
        self.__mu = self.__sinmod.get_data_at_coordinates(coordinates)[:, -1].reshape(-1, 1)

    def assimilate_data(self, dataset: np.ndarray) -> None:
        """
        Assimilate dataset to GRF kernel.

        Args:
            dataset: np.array([x, y, z, sal])

        Methodology:
            1. Find the index using KDTree.
            2. Average the values to each cell.
            3. Update the kernel mean and covariance matrix.

        Returns:
            None

        Examples:
            >>> dataset = np.array([[0, 0, 0], [1, 1, 1]])
            >>> grf = GRF()
            >>> grf.assimilate_data(dataset)
            >>> grf.get_mu()
        """
        df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'salinity'])
        df.to_csv(self.__foldername_ctd + "D_{:03d}.csv".format(self.__cnt), index=False)
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= self.__MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        distance_min, ind_min_distance = self.__grf_grid_kdtree.query(dataset[:, :3])
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated)

        data = np.hstack((ind_assimilated.reshape(-1, 1), salinity_assimilated))
        df = pd.DataFrame(data, columns=['ind', 'salinity'])
        df.to_csv(self.__foldername + "D_{:03d}.csv".format(self.__cnt), index=False)
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")
        self.__cnt += 1

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray) -> None:
        """
        Update GRF kernel based on sampled data.

        Args:
            ind_measured: indices where the data is assimilated.
            salinity_measured: measurements at sampeld locations, dimension: m x 1

        Methodology:
            1. Loop through each measurement and construct the measurement matrix F.
            2. Construct the measurement noise matrix R.
            3. Update the kernel mean and covariance matrix using
                .. math::

                    \mu = \mu + \Sigma F^T (F \Sigma F^T + R)^{-1} (y - F \mu)

                    \Sigma = \Sigma - \Sigma F^T (F \Sigma F^T + R)^{-1} F \Sigma

        Returns:
            None

        Examples:
            >>> ind_measured = np.array([0, 1])
            >>> salinity_measured = np.array([0, 1])
            >>> grf = GRF()
            >>> grf.__update(ind_measured, salinity_measured)
            >>> grf.get_mu()

        """
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.__n_grf_grid])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        self.__mu = self.__mu + self.__Sigma @ F.T @ np.linalg.solve(C, (salinity_measured - F @ self.__mu))
        self.__Sigma = self.__Sigma - self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)

    def get_eibv_at_locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Calculate the EIBV for a given set of locations.

        Args:
            locations: np.array([x, y, z])

        Methodology:
            1. Get the indices of the locations.
            2. Calculate the EIBV using the analytical formula with fast approximation.
        """
        indices_candidates = self.__get_ind_from_location(locations)

        pass

    def __get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                return self.__grf_grid_kdtree.query(loc.reshape(1, -1))[1]
            elif dm == 2:
                return self.__grf_grid_kdtree.query(loc)[1]
            else:
                return None
        else:
            return None

    def __get_posterior_variance(self, ind: np.ndarray) -> np.ndarray:
        msamples = ind.shape[0]
        F = np.zeros([msamples, self.__n_grf_grid])
        for i in range(msamples):
            F[i, ind[i]] = True
        R = np.eye(msamples) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        Sigma_posterior = self.__Sigma - self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)

        pass

    def __get_eibv(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

        Args:
            mu: n x 1 dimension
            sigma_diag: n x 1 dimension
            vr_diag: n x 1 dimension

        Methodology:
            1. Calculate the probability of exceedance of the threshold using a bivariate cumulative dentisty function.
                .. math::
                    p = \\Phi(\\frac{\\theta - \mu}{\sigma}) - \\Phi(\\frac{\\theta - \mu}{\sigma}) \\Phi(\\frac{\\theta - \mu}{\sigma})

            2. Calculate eibv by summing up the product of p*(1-p).

        Returns:
            eibv: information based on variance reduction, dimension: n x 1

        Examples:
            >>> grf = GRF()
            >>> eibv = grf.__get_eibv(grf.__mu, grf.__sigma_diag, grf.__vr_diag)

        """
        eibv = .0
        for i in range(len(mu)):
            sn2 = sigma_diag[i]
            vn2 = vr_diag[i]

            sn = np.sqrt(sn2)
            m = mu[i]

            mur = (self.__threshold - m) / sn

            sig2r_1 = sn2 + vn2
            sig2r = vn2

            eibv += multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                            np.array([[sig2r_1, -sig2r],
                                                      [-sig2r, sig2r_1]]).squeeze())
        return eibv

    def set_sigma(self, value: float) -> None:
        """
        Set space variability.

        Args:
            value: space variability

        Examples:
            >>> grf = GRF()
            >>> grf.set_sigma(0.1)

        """
        self.__sigma = value

    def set_lateral_range(self, value: float) -> None:
        """
        Set lateral range.

        Args:
            value: lateral range

        Examples:
            >>> grf = GRF()
            >>> grf.set_lateral_range(0.1)

        """
        self.__lateral_range = value

    def set_nugget(self, value: float) -> None:
        """
        Set nugget.

        Args:
            value: nugget

        Examples:
            >>> grf = GRF()
            >>> grf.set_nugget(0.1)

        """
        self.__nugget = value

    def set_threshold(self, value: float) -> None:
        """
        Set threshold.

        Args:
            value: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.set_threshold(0.1)

        """
        self.__threshold = value

    def set_mu(self, value: np.ndarray) -> None:
        """
        Set mean of the field.

        Args:
            value: mean of the field

        Examples:
            >>> grf = GRF()
            >>> grf.set_mu(np.array([0.1, 0.2, 0.3]))

        """
        self.__mu = value

    def get_sigma(self) -> float:
        """
        Return variability of the field.

        Returns:
            sigma: space variability

        Examples:
            >>> grf = GRF()
            >>> grf.get_sigma()
            1.0
        """
        return self.__sigma

    def get_lateral_range(self) -> float:
        """
        Return lateral range.

        Returns:
            lateral_range: lateral range

        Examples:
            >>> grf = GRF()
            >>> grf.get_lateral_range()
            600.0
        """
        return self.__lateral_range

    def get_nugget(self) -> float:
        """
        Return nugget of the field.

        Returns:
            nugget: nugget

        Examples:
            >>> grf = GRF()
            >>> grf.get_nugget()
            0.0

        """
        return self.__nugget

    def get_threshold(self) -> float:
        """
        Return threshold.

        Returns:
            threshold: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.get_threshold()
            27.0

        """
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        """
        Return mean vector.

        Returns:
            mu: mean vector

        Examples:
            >>> grf = GRF()
            >>> grf.get_mu()
            array([0.1, 0.2, 0.3])

        """
        return self.__mu

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return Covariance.

        Returns:
            Sigma: Covariance matrix

        Examples:
            >>> grf = GRF()
            >>> grf.get_covariance_matrix()
            array([[1.00000000e+00, 9.99999998e-01, 9.99999994e-01],
                   [9.99999998e-01, 1.00000000e+00, 9.99999998e-01],
                   [9.99999994e-01, 9.99999998e-01, 1.00000000e+00]])

        """
        return self.__Sigma


if __name__ == "__main__":
    g = GRF()
