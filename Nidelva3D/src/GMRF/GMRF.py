"""
This helper solves all the essential problems associated with GMRF class.
"""
from GMRF.spde import spde
from typing import Union
from WGS import WGS
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
import os
import pandas as pd
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from usr_func.checkfolder import checkfolder
import time


class GMRF:
    __MIN_DEPTH_FOR_DATA_ASSIMILATION = .25
    __GMRF_DISTANCE_NEIGHBOUR = 32
    __gmrf_grid = None
    __N_gmrf_grid = 0
    __rotated_angle = .0
    __cnt = 0

    def __init__(self):
        self.__spde = spde()
        self.construct_gmrf_grid()
        t = time.time()
        f = os.getcwd()
        self.foldername = f + "/GMRF/data/{:d}/".format(int(t))
        self.foldername_ctd = f + "/GMRF/raw_ctd/{:d}".format(int(t))
        checkfolder(self.foldername)

    def construct_gmrf_grid(self) -> None:
        """
        Construct GMRF grid by converting lats, lons to xy.
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

    def assimilate_data(self, dataset: np.ndarray) -> tuple:
        """
        Assimilate dataset to spde kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, z, sal])
        """
        df = pd.DataFrame(dataset, columns=['x', 'y', 'z', 'salinity'])
        df.to_csv(self.foldername_ctd + "D_{:03d}.csv".format(self.__cnt))
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
        data = np.hstack((ind_assimilated.reshape(-1, 1), salinity_assimilated))
        df = pd.DataFrame(data, columns=['ind', 'salinity'])
        df.to_csv(self.foldername + "D_{:03d}.csv".format(self.__cnt))
        self.__cnt += 1
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1)
        return ind_assimilated, salinity_assimilated, ind_min_distance

    def get_eibv_at_locations(self, loc: np.ndarray) -> np.ndarray:
        """
        Get EIBV at candidate locations.

        Args:
            loc: np.array([[x1, y1, z1],
                           [x2, y2, z2],
                           ...
                           [xn, yn, zn]])

        Returns:
            EIBV associated with each location.
        """
        # s1: get indices
        id = self.get_ind_from_location(loc)
        # s2: get post variance from spde
        post_var = self.__spde.candidate(ks=id)
        # s3: get eibv using post variance
        eibv = []
        for i in range(len(id)):
            ibv = GMRF.get_ibv(self.__spde.threshold, self.__spde.mu, post_var[:, i])
            eibv.append(ibv)
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

    @staticmethod
    def get_ibv(threshold: float, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
        """
        Calculate the integrated bernoulli variance given mean and variance.
        """
        p = norm.cdf(threshold, mu, sigma_diag)
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

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


