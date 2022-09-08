"""
SINMOD handles data extraction from SINMOD data files. It loads the latest SINMOD datafile from a bunch of netCDF4 files
Then the data will be extracted to the desired coordinates by using minimum distance assignment.
"""

from WGS import WGS
from Planner.Myopic3D import Myopic3D
import os
import re
import numpy as np
import netCDF4
from datetime import datetime
import time
import pandas as pd


class SINMOD:
    """
    SINMOD handler
    """
    __SINMOD_MAX_DEPTH_LAYER = 10
    __sinmod_filepath = os.getcwd() + "/../../../../Data/Nidelva/SINMOD_DATA/samples/"
    __sinmod_files = os.listdir(__sinmod_filepath)
    __sinmod_files.sort()
    __sinmod_file = __sinmod_files[-1]  # always get the latest file
    __sinmod_data = netCDF4.Dataset(__sinmod_filepath + __sinmod_file)

    __ind_before = re.search("samples_", __sinmod_file)
    __ind_after = re.search(".nc", __sinmod_file)
    __date_string = __sinmod_file[__ind_before.end():__ind_after.start()]
    __ref_timestamp = datetime.strptime(__date_string, "%Y.%m.%d").timestamp()
    __timestamp = np.array(__sinmod_data["time"]) * 24 * 3600 + __ref_timestamp  # change ref timestamp

    __lat_sinmod = np.array(__sinmod_data['gridLats'])
    __lon_sinmod = np.array(__sinmod_data['gridLons'])
    __depth_sinmod = np.array(__sinmod_data['zc'])[:__SINMOD_MAX_DEPTH_LAYER]

    __salinity_time_ave = np.mean(np.array(__sinmod_data['salinity'])[:, :__SINMOD_MAX_DEPTH_LAYER, :, :], axis=0)
    __df_sinmod = []
    for i in range(__lat_sinmod.shape[0]):
        for j in range(__lat_sinmod.shape[1]):
            for k in range(len(__depth_sinmod)):
                __df_sinmod.append([__lat_sinmod[i, j], __lon_sinmod[i, j],
                                    __depth_sinmod[k], __salinity_time_ave[k, i, j]])
    __df_sinmod = np.array(__df_sinmod)

    @staticmethod
    def __vectorize(value: np.ndarray) -> np.ndarray:
        """
        Reshape value to a column vector.
        Args:
            value: flattened array with shape (n,)
        Returns:
            vectorized array with shape (n, 1)
        """
        return value.reshape(-1, 1)

    @staticmethod
    def __get_distance_matrix(s1, s2):
        """
        Return distance matrix between two column vectors.
        Args:
            s1: site one
            s2: site two
        Returns:
            Distance matrix from site one to site two.
        """
        s1 = np.array(s1).reshape(-1, 1)
        s2 = np.array(s2).reshape(-1, 1)
        dx = np.dot(s1, np.ones([1, len(s2)]))
        dy = np.dot(np.ones([len(s1), 1]), s2.T)
        return dx - dy

    def get_data_at_coordinates(self, coordinates: np.array) -> np.ndarray:
        """
        Get sinmod data from coordinates.
        Args:
            coordinates: np.array([[lat1, lon1, depth1],
                                   [lat2, lon2, depth2],
                                   ...
                                   [latn, lonn, depthn]])
        Returns:
            data frame containing interpolated data values and coordinates.
            np.array([[lat1, lon1, depth1, sal1],
                      [lat2, lon2, depth2, sal2],
                      ...
                      [latn, lonn, depthn, saln]])
        """
        lat_sinmod = self.__df_sinmod[:, 0]
        lon_sinmod = self.__df_sinmod[:, 1]
        depth_sinmod = self.__df_sinmod[:, 2]
        salinity_sinmod = self.__df_sinmod[:, 3]

        print("Coordinates shape: ", coordinates.shape)
        lat_coordinates = coordinates[:, 0]
        lon_coordinates = coordinates[:, 1]
        depth_coordinates = coordinates[:, 2]
        ts = time.time()
        x_coordinates, y_coordinates = WGS.latlon2xy(lat_coordinates, lon_coordinates)
        x_sinmod, y_sinmod = WGS.latlon2xy(lat_sinmod, lon_sinmod)
        x_coordinates, y_coordinates, depth_coordinates, x_sinmod, y_sinmod, depth_sinmod = \
            map(self.__vectorize, [x_coordinates, y_coordinates, depth_coordinates,
                                   x_sinmod, y_sinmod, depth_sinmod])

        t1 = time.time()
        DistanceMatrix_x = self.__get_distance_matrix(x_coordinates, x_sinmod)
        t2 = time.time()
        print("Distance matrix - x finished, time consumed: ", t2 - t1)

        t1 = time.time()
        DistanceMatrix_y = self.__get_distance_matrix(y_coordinates, y_sinmod)
        t2 = time.time()
        print("Distance matrix - y finished, time consumed: ", t2 - t1)

        t1 = time.time()
        DistanceMatrix_depth = self.__get_distance_matrix(depth_coordinates, depth_sinmod)
        t2 = time.time()
        print("Distance matrix - depth finished, time consumed: ", t2 - t1)

        t1 = time.time()
        DistanceMatrix = DistanceMatrix_x ** 2 + DistanceMatrix_y ** 2 + DistanceMatrix_depth ** 2
        t2 = time.time()
        print("Distance matrix - total finished, time consumed: ", t2 - t1)

        t1 = time.time()
        ind_interpolated = np.argmin(DistanceMatrix, axis=1) # interpolated vectorised indices
        t2 = time.time()
        print("Interpolation finished, time consumed: ", t2 - t1)

        sal_interpolated = salinity_sinmod[ind_interpolated].reshape(-1, 1)
        df_interpolated = np.hstack((coordinates, sal_interpolated))
        t2 = time.time()
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated


if __name__ == "__main__":
    s = SINMOD()
    my = Myopic3D()
    grid = my.gmrf.get_gmrf_grid()
    lat, lon = WGS.xy2latlon(grid[:, 0], grid[:, 1])
    grid_wgs = np.vstack((lat, lon, grid[:, 2])).T
    N = 5000
    for i in np.arange(0, len(grid_wgs), N):
        print(i)
        g = grid_wgs[i:i+N, :]
        d = s.get_data_at_coordinates(g)
        ddf = pd.DataFrame(d, columns=['lat', 'lon', 'depth', 'salinity'])
        ddf.to_csv("/Users/yaolin/Downloads/data/df_{:05d}.csv".format(i), index=False)
        # df = np.append(df, d, axis=0)

    path = "/Users/yaolin/Downloads/data/"
    files = os.listdir(path)
    dt = np.empty([0, 4])
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(path + file)
            dt = np.append(dt, df.to_numpy(), axis=0)

    dtf = pd.DataFrame(dt, columns=['lat', 'lon', 'depth', 'salinity'])
    dtf.to_csv(path + "total/data.csv", index=False)

    x, y = WGS.latlon2xy(dt[:, 0], dt[:, 1])
    dn = np.vstack((x, y, dt[:, 2], dt[:, -1])).T
    dnf = pd.DataFrame(dn, columns=['x', 'y', 'z', 'salinity'])
    dnf.to_csv(path + "total/data_ned.csv", index=False)

    import matplotlib.pyplot as plt
    import plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Scatter3d(
        x=dt[:, 1],
        y=dt[:, 0],
        z=-dt[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=dt[:, -1],  # set color to an array/list of desired values
            colorscale='RdBu',  # choose a colorscale
            showscale=True,
        ),
    )])

    figpath = os.getcwd() + "/../fig/GroundTruth.html"
    plotly.offline.plot(fig, filename=figpath, auto_open=True)


# import os
# import pandas as pd
# import plotly
# import plotly.graph_objects as go
#
# dt = pd.read_csv(os.getcwd() + "/AUVSimulator/simulated_truth_wgs.csv").to_numpy()
# x, y = WGS.latlon2xy(dt[:, 0], dt[:, 1])
# dn = np.vstack((x, y, dt[:, 2], dt[:, -1])).T
# dnf = pd.DataFrame(dn, columns=['x', 'y', 'z', 'salinity'])
# dnf.to_csv(os.getcwd() + "/AUVSimulator/simulated_truth.csv", index=False)
#
#
# fig = go.Figure(data=[go.Scatter3d(
#     x=dt[:, 1],
#     y=dt[:, 0],
#     z=-dt[:, 2],
#     mode='markers',
#     marker=dict(
#         size=4,
#         color=dt[:, -1],  # set color to an array/list of desired values
#         colorscale='RdBu',  # choose a colorscale
#         showscale=True,
#     ),
# )])
#
# figpath = os.getcwd() + "/../fig/GroundTruth.html"
# plotly.offline.plot(fig, filename=figpath, auto_open=True)




