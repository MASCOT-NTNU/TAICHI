"""
SINMOD module handles the data interpolation for a given set of coordinates.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

Methodology:
    1. Read SINMOD data from netCDF file.
    2. Construct KDTree for the SINMOD grid.
    3. For a given set of coordinates, find the nearest SINMOD grid point.
    4. Interpolate the data using the nearest SINMOD grid point.
"""
from WGS import WGS
from pykdtree.kdtree import KDTree
import xarray as xr
import re
import numpy as np
import netCDF4
from datetime import datetime
import time


class SINMOD:
    """
    SINMOD class handles the data interpolation for a given set of coordinates.
    """
    def __init__(self, filepath: str = None) -> None:
        if filepath is None:
            raise ValueError("Please provide the filepath to SINMOD data.")
        else:
            self.sinmod_filepath = filepath
            self.sinmod_data = netCDF4.Dataset(self.sinmod_filepath)
            ind_before = re.search("samples_", self.sinmod_filepath)
            ind_after = re.search(".nc", self.sinmod_filepath)
            date_string = self.sinmod_filepath[ind_before.end():ind_after.start()]
            ref_timestamp = datetime.strptime(date_string, "%Y.%m.%d").timestamp()
            self.timestamp = np.array(self.sinmod_data["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp

            self.lat_sinmod = np.array(self.sinmod_data['gridLats'])
            self.lon_sinmod = np.array(self.sinmod_data['gridLons'])
            self.x_sinmod, self.y_sinmod = WGS.latlon2xy(self.lat_sinmod, self.lon_sinmod)
            self.depth_sinmod = np.array(self.sinmod_data['zc'])
            self.salinity_sinmod = np.array(self.sinmod_data['salinity'])
            salinity_sinmod_time_ave = np.mean(self.salinity_sinmod[:, :, :, :], axis=0)
            self.data_sinmod = []
            for i in range(self.lat_sinmod.shape[0]):
                for j in range(self.lat_sinmod.shape[1]):
                    for k in range(len(self.depth_sinmod)):
                        self.data_sinmod.append([self.x_sinmod[i, j], self.y_sinmod[i, j],
                                            self.depth_sinmod[k], salinity_sinmod_time_ave[k, i, j]])
            self.data_sinmod = np.array(self.data_sinmod)
            t1 = time.time()
            self.sinmod_grid_tree = KDTree(self.data_sinmod[:, :3])
            t2 = time.time()
            print("KDTree construction time: ", t2 - t1)

            t1 = time.time()
            x_da = xr.DataArray(self.x_sinmod, dims=['y', 'x'])
            y_da = xr.DataArray(self.y_sinmod, dims=['y', 'x'])
            depth_da = xr.DataArray(self.depth_sinmod, dims=['depth'])
            timestamp_da = xr.DataArray(self.timestamp, dims=['time'])
            salinity_da = xr.DataArray(self.salinity_sinmod, dims=['time', 'depth', 'y', 'x'])
            ds = xr.Dataset({"salinity": salinity_da, "x_da": x_da, "y_da": y_da, "d_da": depth_da,
                             "time_da": timestamp_da})

            ds.sel()
            # df_sinmod_timestamp = {}
            # for i in range(len(self.timestamp)):
            #     df_sinmod_timestamp[self.timestamp[i]] = {}
            #     for j in range(self.lat_sinmod.shape[0]):
            #         for k in range(self.lat_sinmod.shape[1]):
            #             for l in range(len(self.depth_sinmod)):
            #                 df_sinmod_timestamp[self.timestamp[i]] = ([self.timestamp[i], self.lat_sinmod[j, k],
            #                                                            self.lon_sinmod[j, k], self.depth_sinmod[l],
            #                                                            self.salinity_sinmod[i, l, j, k]])
            # df_sinmod_timestamp = np.array(df_sinmod_timestamp)
            # x, y = WGS.latlon2xy(df_sinmod_timestamp[:, 1], df_sinmod_timestamp[:, 2])
            # self.data_sinmod_timestamp = np.hstack((df_sinmod_timestamp[:, 0].reshape(-1, 1), x.reshape(-1, 1),
            #                                         y.reshape(-1, 1), df_sinmod_timestamp[:, 3:]))
            # self.sinmod_timestamp_kdtree = KDTree(self.data_sinmod_timestamp[:, :-1])
            # t2 = time.time()
            # print("KDTree construction time for the timestamp inclusion: ", t2 - t1)

    def get_data_at_locations(self, locations: np.array) -> np.ndarray:
        """
        Get SINMOD data values at given locations.

        Args:
            location: x, y, depth coordinates
            Example: np.array([[x1, y1, depth1],
                               [x2, y2, depth2],
                               ...
                               [xn, yn, depthn]])
        Returns:
            SINMOD data values at given locations.
        """
        ts = time.time()
        dist, ind = self.sinmod_grid_tree.query(locations.astype(np.float32))
        sal_interpolated = self.data_sinmod[ind, -1].reshape(-1, 1)
        df_interpolated = np.hstack((locations, sal_interpolated))
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated

    def get_data_at_timestamp_and_locations(self, timestamp: float, time_locations: np.array) -> np.ndarray:
        """
        Get SINMOD data values at given locations and timestamp.

        Args:
            timestamp: timestamp in seconds
            time_locations: x, y, depth coordinates
            Example: np.array([[x1, y1, depth1],
                               [x2, y2, depth2],
                               ...
                               [xn, yn, depthn]])
        Returns:
            SINMOD data values at given locations and timestamp.
        """
        ind_timestamp = np.argmin(np.abs(self.data_sinmod_timestamp[:, 0] - timestamp))
        salinity_sinmod_at_timestamp = self.salinity_sinmod[ind_timestamp, :, :, :]
        ts = time.time()
        dist, ind = self.sinmod_timestamp_kdtree.query(time_locations.astype(np.float32))
        sal_interpolated = self.data_sinmod_timestamp[ind, -1].reshape(-1, 1)
        df_interpolated = np.hstack((time_locations, sal_interpolated))
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated

    def get_data(self) -> np.ndarray:
        """
        Return the dataset of SINMOD data.
        """
        return self.data_sinmod


if __name__ == "__main__":
    s = SINMOD()
