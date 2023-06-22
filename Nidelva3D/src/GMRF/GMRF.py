"""
GMRF wraps the spde class and provides a higher level interface for the GMRF model.

Created on 2023-05-22
Author: Yaolin Ge
Email: geyaolin@gmail.com

Functionality:
    1. Save the data in folders.
    2. Assimilate the in-situ data.
    3. Calculate the EIBV for given locations.
        - Analytical solver, relatively slow.
        - Approximate solver, fast.

Notes:
    1. The EIBV calculation is based on the approximation of the bivariate normal distribution.
    2. The bivariate CDF is pre-calculated and stored in a joblib interpolater.
    3. Details should be checked in `cal_cdf_regular_grid_interpolator.py`.

"""
from WGS import WGS
from GMRF.spde import spde
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from usr_func.checkfolder import checkfolder
from typing import Union
import numpy as np
from scipy.stats import multivariate_normal
from pykdtree.kdtree import KDTree
import os
import pandas as pd
import time
import joblib
from scipy.interpolate import RegularGridInterpolator


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
        self.__load_cdf_interpolator()

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
        self.__gmrf_grid_kdtree = KDTree(self.__gmrf_grid)

        """
        Get the rotation of the grid, used for later plotting.
        """
        box = np.load(filepath + "grid.npy")
        polygon = box[:, 2:]
        polygon = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
        polygon = sort_polygon_vertices(polygon)
        self.__rotated_angle = np.math.atan2(polygon[1, 0] - polygon[0, 0],
                                             polygon[1, 1] - polygon[0, 1])

    def __load_cdf_interpolator(self) -> None:
        t1 = time.time()
        self.__rho_values, self.__z1_values, self.__z2_values, self.__cdf_values = joblib.load(
            os.getcwd() + "/GMRF/interpolator_medium.joblib")
        self.__interpolators = [
            RegularGridInterpolator((self.__z1_values, self.__z2_values), self.__cdf_values[i, :, :],
                                    bounds_error=False, fill_value=None) for i in range(self.__rho_values.size)]
        t2 = time.time()
        print("Loading interpolators finished, time cost: {:.2f} s".format(t2 - t1))

    def __query_cdf(self, rho, z1, z2) -> np.ndarray:
        # s1, Find the index of the closest rho layer
        i = np.abs(self.__rho_values - rho).argmin()
        # s2, Use the interpolator for this layer to interpolate the value
        return self.__interpolators[i]([[z1, z2]])

    def assimilate_data(self, dataset: np.ndarray) -> tuple:
        """
        Assimilate dataset to spde kernel.

        Args:
            dataset: np.array([x, y, z, sal])

        Methodology:
            1. Remove the noise layer.
            2. Compute the distance matrix between gmrf grid and dataset grid.
            3. Update the spde kernel.
        """
        # ss1: save raw ctd
        df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'salinity'])
        df.to_csv(self.__foldername_ctd + "D_{:03d}.csv".format(self.__cnt), index=False)
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= self.__MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        distance_min, ind_min_distance = self.__gmrf_grid_kdtree.query(dataset[:, :3], k=1)
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
        prior_variance = self.__spde.mvar()

        eibv = []
        # eibv2 = []
        for i in range(len(indices_candidates)):
            variance_reduction = prior_variance - marginal_variance[:, i]
            mu_input = self.__spde.mu
            sigma_diag_input = marginal_variance[:, i]
            vr_diag_input = variance_reduction
            threshold = self.__spde.threshold

            # eibv_temp1, EBV1 = self.__get_eibv_analytical(mu=mu_input, sigma_diag=sigma_diag_input,
            #                                                vr_diag=vr_diag_input, threshold=threshold)
            eibv_temp = self.__get_eibv_analytical_fast(mu=mu_input, sigma_diag=sigma_diag_input,
                                                        vr_diag=vr_diag_input, threshold=threshold)
            eibv.append(eibv_temp)
            # eibv_temp2 = self.__get_eibv_analytical(mu=mu_input, sigma_diag=sigma_diag_input, vr_diag=vr_diag_input,
            #                                         threshold=threshold)
            # eibv2.append(eibv_temp2)
        return np.array(eibv)

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
        EBV = []
        sn2 = sigma_diag
        vn2 = vr_diag
        sn = np.sqrt(sn2)

        mur = (threshold - mu) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1
        for i in range(len(z1)):
            ebv = multivariate_normal.cdf([z1[i], z2[i]], mean=[0, 0], cov=[[1, rho[i]], [rho[i], 1]])
            eibv += ebv
            EBV.append(ebv)
        return eibv

    def __get_eibv_analytical_fast(self, mu: np.ndarray, sigma_diag: np.ndarray,
                                   vr_diag: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate the eibv using the analytical formula but using a loaded cdf dataset.
        """
        # s1, calculate the z1, z2, rho
        sn2 = sigma_diag
        vn2 = vr_diag
        sn = np.sqrt(sn2)
        mur = (np.ones_like(mu) * threshold - mu) / sn
        sig2r_1 = sn2 + vn2
        sig2r = vn2
        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1
        grid = np.stack((rho, z1, z2), axis=1)

        # s2, query the cdf from the loaded interpolators
        ebv = np.array([self.__query_cdf(rho, z1, z2) for rho, z1, z2 in grid])
        ebv[ebv < 0] = 0
        eibv = np.sum(ebv)
        return eibv

    def get_ind_from_location(self, loc: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            loc: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        if len(loc) > 0:
            dm = loc.ndim
            if dm == 1:
                return self.__gmrf_grid_kdtree.query(loc.reshape(1, -1))[1]
            elif dm == 2:
                return self.__gmrf_grid_kdtree.query(loc)[1]
            else:
                return None
        else:
            return None

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

    def get_grid(self) -> np.ndarray:
        """
        Returns: gmrf_grid (private variable)
        """
        return self.__gmrf_grid

    def get_rotated_angle(self) -> float:
        """
        Returns: rotated angle of the gmrf grid.
        """
        return self.__rotated_angle

    def get_mu(self) -> np.ndarray:
        """
        Returns: conditional mean of the GMRF field.
        """
        return self.__spde.mu

    def get_mvar(self) -> np.ndarray:
        """
        Returns: conditional mariginal variance of the GMRF field.
        """
        return self.__spde.mvar()

    def get_threshold(self) -> float:
        """
        Returns: threshold for the EIBV calculation.
        """
        return self.__spde.threshold

    def interpolate_mu4locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Interpolate the conditional mean of the GMRF field at given locations.
        """
        return self.__spde.mu[self.__gmrf_grid_kdtree.query(locations)[1]]

    def get_spde(self) -> 'SPDE':
        """ Return SPDE """
        return self.__spde


if __name__ == "__main__":
    s = GMRF()
