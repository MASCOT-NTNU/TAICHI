"""
GMRF wraps the spde class and provides a higher level interface for the GMRF model.

Created on 2023-05-22
Author: Yaolin Ge
Email: geyaolin@gmail.com


"""
from WGS import WGS
from GMRF.spde import spde
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from usr_func.checkfolder import checkfolder
from typing import Union
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from numba import njit, jit
import os
import pandas as pd
import time


class GMRF:
    """
    GMRF (Gaussian Markov Random Field) class for data assimilation.



    Attributes:
        __MIN_DEPTH_FOR_DATA_ASSIMILATION: Minimum depth for data assimilation.
        __GMRF_DISTANCE_NEIGHBOUR: Distance for GMRF neighbour.
        __gmrf_grid: GMRF grid.
        __N_gmrf_grid: Number of GMRF grid.
        __rotated_angle: Rotated angle.
        __cnt: Count.

    """
    __MIN_DEPTH_FOR_DATA_ASSIMILATION = .25
    __GMRF_DISTANCE_NEIGHBOUR = 32
    __gmrf_grid = None
    __N_gmrf_grid = 0
    __rotated_angle = .0
    __cnt = 0

    def __init__(self) -> None:
        self.__spde = spde()
        self.__create_data_folders()
        self.__construct_gmrf_grid()
        self.__load_cdf_table()

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
        self.__foldername = f + "/GMRF/data/{:d}/".format(t)
        self.__foldername_ctd = f + "/GMRF/raw_ctd/{:d}/".format(t)
        self.__foldername_thres = f + "/GMRF/threshold/{:d}/".format(t)
        checkfolder(self.__foldername)
        checkfolder(self.__foldername_ctd)
        checkfolder(self.__foldername_thres)

    def __construct_gmrf_grid(self) -> None:
        """
        Construct the GMRF grid.

        Method:
            1. Load the grid from the file.
            2. Convert the lat/lon to x/y.
            3. Stack the x/y/z to a 3D array.

        """
        filepath = os.getcwd() + "/GMRF/models/"
        lat = np.load(filepath + "lats.npy")
        lon = np.load(filepath + "lons.npy")
        depth = np.load(filepath + "depth.npy")
        x, y = WGS.latlon2xy(lat, lon)
        z = depth
        self.__gmrf_grid = np.stack((x, y, z), axis=1)
        self.__N_gmrf_grid = self.__gmrf_grid.shape[0]

        """
        Get the rotation of the grid, used for later plotting.
        """
        box = np.load(filepath + "grid.npy")
        polygon = box[:, 2:]
        polygon = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
        polygon = sort_polygon_vertices(polygon)
        self.__rotated_angle = np.math.atan2(polygon[1, 0] - polygon[0, 0],
                                             polygon[1, 1] - polygon[0, 1])

    def __load_cdf_table(self) -> None:
        """
        Load cdf table for the analytical solution.
        """
        table = np.load("./GMRF/cdf.npz")
        self.__cdf_z1 = table["z1"]
        self.__cdf_z2 = table["z2"]
        self.__cdf_rho = table["rho"]
        self.__cdf_table = table["cdf"]

    def assimilate_data(self, dataset: np.ndarray) -> tuple:
        """
        Assimilate dataset to spde kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, z, sal])
        """
        # ss1: save raw ctd
        df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'salinity'])
        df.to_csv(self.__foldername_ctd + "D_{:03d}.csv".format(self.__cnt))
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= self.__MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        xd = dataset[:, 0].reshape(-1, 1)
        yd = dataset[:, 1].reshape(-1, 1)
        zd = dataset[:, 2].reshape(-1, 1)
        Fgmrf = np.ones([1, self.__N_gmrf_grid])
        Fdata = np.ones([dataset.shape[0], 1])
        xg = self.__gmrf_grid[:, 0].reshape(-1, 1)
        yg = self.__gmrf_grid[:, 1].reshape(-1, 1)
        zg = self.__gmrf_grid[:, 2].reshape(-1, 1)
        # t1 = time.time()
        dx = (xd @ Fgmrf - Fdata @ xg.T) ** 2
        dy = (yd @ Fgmrf - Fdata @ yg.T) ** 2
        dz = ((zd @ Fgmrf - Fdata @ zg.T) * self.__GMRF_DISTANCE_NEIGHBOUR) ** 2
        dist = dx + dy + dz
        ind_min_distance = np.argmin(dist, axis=1)  # used only for unittest.
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        self.__spde.update(rel=salinity_assimilated, ks=ind_assimilated)

        # ss2: save assimilated data
        data = np.hstack((ind_assimilated.reshape(-1, 1), salinity_assimilated))
        df = pd.DataFrame(data, columns=['ind', 'salinity'])
        df.to_csv(self.__foldername + "D_{:03d}.csv".format(self.__cnt))

        # ss3: save threshold
        threshold = np.array(self.__spde.getThreshold())
        df = pd.DataFrame(threshold.reshape(1, -1), columns=['threshold'])
        df.to_csv(self.__foldername_thres + "D_{:03d}.csv".format(self.__cnt))

        self.__cnt += 1
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1)
        return ind_assimilated, salinity_assimilated, ind_min_distance

    def get_eibv_at_locations(self, loc: np.ndarray) -> np.ndarray:
        """
        Get EIBV at given locations.

        Method:
            1. Get the indices of the locations.
            2. Get the marginal variance of the field.
            3. Get the posterior marginal variance of the field.
            4. Calculate the EIBV.

        Args:
            loc: np.array([[x1, y1, z1],
                           [x2, y2, z2],
                           ...
                           [xn, yn, zn]])

        Returns:
            EIBV associated with given locations.
        """
        # s1: get indices of the locations
        indices_candidates = self.get_ind_from_location(loc)
        marginal_variance = self.__spde.candidate(ks=indices_candidates)

        eibv = []
        for i in range(len(indices_candidates)):
            variance_reduction = self.__spde.mvar() - marginal_variance[:, i]
            eibv_temp1 = self.__get_eibv_analytical(mu=self.__spde.mu, sigma_diag=marginal_variance[:, i],
                                                    vr_diag=variance_reduction)
            eibv_temp2 = self.__get_eibv_analytical_fast(mu=self.__spde.mu, sigma_diag=marginal_variance[:, i],
                                                         vr_diag=variance_reduction)
            print("eibv_temp1: ", eibv_temp1)
            print("eibv_temp2: ", eibv_temp2)
            eibv.append(eibv_temp1)

            # eibv.append(ibv)
        return np.array(eibv)

    def get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                d = cdist(self.__gmrf_grid, loc.reshape(1, -1))
                return np.argmin(d, axis=0)
            elif dm == 2:
                d = cdist(self.__gmrf_grid, loc)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def __get_eibv_analytical(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
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
        for i in range(len(mu)):
            sn2 = sigma_diag[i]
            vn2 = vr_diag[i]

            sn = np.sqrt(sn2)
            m = mu[i]

            mur = (self.__spde.threshold - m) / sn

            sig2r_1 = sn2 + vn2
            sig2r = vn2

            eibv += multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                            np.array([[sig2r_1, -sig2r],
                                                      [-sig2r, sig2r_1]]).squeeze())
        return eibv

    @jit
    def __get_eibv_analytical_fast(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula but using a loaded cdf dataset.
        """
        eibv = .0
        for i in range(len(mu)):
            sn2 = sigma_diag[i]
            vn2 = vr_diag[i]

            sn = np.sqrt(sn2)
            m = mu[i]

            mur = (self.__spde.threshold - m) / sn

            sig2r_1 = sn2 + vn2
            sig2r = vn2

            z1 = mur
            z2 = -mur
            rho = -sig2r / sig2r_1

            ind1 = np.argmin(np.abs(z1 - self.__cdf_z1))
            ind2 = np.argmin(np.abs(z2 - self.__cdf_z2))
            ind3 = np.argmin(np.abs(rho - self.__cdf_rho))
            eibv += self.__cdf_table[ind1][ind2][ind3]
        return eibv

    def get_location_from_ind(self, ind: Union[int, list]) -> np.ndarray:
        """
        Get that location given its index in the gmrf grid.
        Args:
            ind: int or list of int specify where the locations are in gmrf grid.
        Returns: numpy array containing locations np.array([[x1, y1, z1],
                                                            [x2, y2, z2],
                                                            ...
                                                            [xn, yn, zn]])
        """
        return self.__gmrf_grid[ind]

    def get_gmrf_grid(self) -> np.ndarray:
        """
        Returns: gmrf_grid (private variable)
        """
        return self.__gmrf_grid

    def get_rotated_angle(self):
        """
        Returns: rotated angle of the gmrf grid.
        """
        return self.__rotated_angle

    def get_mu(self):
        """
        Returns: conditional mean of the GMRF field.
        """
        return self.__spde.mu

    def get_mvar(self):
        """
        Returns: conditional mariginal variance of the GMRF field.
        """
        return self.__spde.mvar()


if __name__ == "__main__":
    s = GMRF()
