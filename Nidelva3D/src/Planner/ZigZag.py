"""
ZigZag plans the path according to pre-defined zigzag patterns.
"""

from WGS import WGS
from usr_func.sort_polygon_vertices import sort_polygon_vertices
import numpy as np
import math
import os


class ZigZag:

    box = np.load(os.getcwd() + "/GMRF/models/grid.npy")
    polygon = box[:, 2:]
    polygon = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
    __polygon_border = sort_polygon_vertices(polygon)
    __rotated_angle = math.atan2(__polygon_border[1, 0] - __polygon_border[0, 0],
                                 __polygon_border[1, 1] - __polygon_border[0, 1])
    __depth_limit = [0.5, 5.5]
    __depth_diff = np.diff(__depth_limit)
    __marginal_distance = 100  # [m] away from the borders.
    __max_pitch = math.radians(10)  # [deg] for AUV.
    __dist_yoyo_lateral = 60  # [m]
    __zigzag_yoyo_path = np.empty([0, 3])

    def __init__(self) -> None:
        self.construct_path()

    def set_polygon_border(self, value: np.ndarray) -> None:
        """
        Set the polygon border.
        Args:
            value: np.array([[x1, y1],
                             [x2, y2]])
        """
        self.__polygon_border = value

    def set_depth_limit(self, value: list) -> None:
        """
        Set the depth limit.
        Args:
            value: [min, max]
        """
        self.__depth_limit = value
        self.__depth_diff = np.abs(np.diff(value))

    def set_maximum_pitch(self, value: float) -> None:
        """
        Set the maximum pitch angle.
        Args:
            value: degrees in float, 30 deg.
        """
        self.__max_pitch = math.radians(value)

    def set_rotated_angle(self, value: float) -> None:
        """
        Set the rotated angle.
        Args:
            value: radians in float, pi/2.
        """
        self.__rotated_angle = value

    def set_marginal_distance(self, value: float) -> None:
        """ Set marginal distance. """
        self.__marginal_distance = value

    def construct_path(self) -> None:
        # s1: rotate polygon back to regular
        RR = np.array([[np.cos(self.__rotated_angle), -np.sin(self.__rotated_angle)],
                       [np.sin(self.__rotated_angle), np.cos(self.__rotated_angle)]])
        polygon = (RR @ self.__polygon_border.T).T

        # s2: get box
        xmin, ymin = map(np.amin, [polygon[:, 0], polygon[:, 1]])
        xmax, ymax = map(np.amax, [polygon[:, 0], polygon[:, 1]])

        # s3: shrink the polygon to produce vertice
        xs = xmin + self.__marginal_distance
        ys = ymin + self.__marginal_distance
        xe = xmax - self.__marginal_distance
        ye = ymax - self.__marginal_distance

        # s4: get yoyo gap distance.
        # self.get_yoyo_lateral_distance()

        # s5: use vertice to connect a path.
        path = np.empty([0, 3])
        order = np.array([[xs, ys],
                          [xe, ye],
                          [xe, ys],
                          [xs, ye],
                          [xs, ys]])

        for i in range(len(order) - 1):
            path = np.append(path, self.get_yoyo_path_between_vertices(order[i], order[i+1]), axis=0)

        # s6: map it back to original coordinate system.
        R = np.array([[np.cos(self.__rotated_angle), np.sin(self.__rotated_angle), 0],
                      [-np.sin(self.__rotated_angle), np.cos(self.__rotated_angle), 0],
                      [0, 0, 1]])
        self.__zigzag_yoyo_path = (R @ path.T).T

    def get_yoyo_path_between_vertices(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Generates a straight line path with yoyo in between
        Args:
            v1: np.array([[x1, y1]])
            v2: np.array([[x2, y2]])
        Returns:
            designed path: np.array([[x, y, z]])
        """
        x1, y1 = v1
        x2, y2 = v2

        # s1: get distance between two vertices.
        dist_lateral = np.sqrt((x1 - x2) ** 2 +
                               (y1 - y2) ** 2)

        # s2: get angle between two vertices.
        angle = math.atan2(x2-x1, y2-y1)

        # s3: get middle points.
        dist_gap = np.arange(0, dist_lateral, self.__dist_yoyo_lateral)
        dist_gap = np.linspace(0, dist_lateral, len(dist_gap))

        # s4: build yoyo path.
        path = []
        for i in range(len(dist_gap)):
            x = x1 + np.sin(angle) * dist_gap[i]
            y = y1 + np.cos(angle) * dist_gap[i]
            if i % 2 == 0:
                z = self.__depth_limit[0]
            else:
                z = self.__depth_limit[1]
            path.append([x, y, z])
        path = np.array(path)
        return path

    def get_polygon_border(self) -> np.ndarray:
        """ Return polygon border. """
        return self.__polygon_border

    def get_depth_limit(self) -> list:
        """ Return depth limit. """
        return self.__depth_limit

    def get_depth_diff(self) -> float:
        """ Return difference between top and bottom depth layer. """
        return self.__depth_diff

    def get_maximum_pitch(self) -> float:
        """ Return maximum pitch. """
        return self.__max_pitch

    def get_yoyo_lateral_distance(self) -> None:
        """ Get yoyo lateral distance. """
        self.__dist_yoyo_lateral = self.__depth_diff / math.tan(self.__max_pitch)

    def get_rotated_angle(self) -> float:
        """ Return rotated angle. """
        return self.__rotated_angle

    def get_marginal_distance(self) -> float:
        """ Return marginal distance. """
        return self.__marginal_distance

    def get_zigzag_path(self):
        """ Return constructed zigzag path. """
        return self.__zigzag_yoyo_path


if __name__ == "__main__":
    z = ZigZag()
    z.construct_path()
    path = z.get_zigzag_path()
    lat, lon = WGS.xy2latlon(path[:, 0], path[:, 1])
    wp = np.vstack((lat, lon, path[:, 2])).T
    import matplotlib.pyplot as plt
    plt.plot(wp[:, 1], wp[:, 0], 'k.-')
    plt.show()

    import plotly
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Scatter3d(
        x=wp[:, 1],
        y=wp[:, 0],
        z=-wp[:, 2],
        mode="markers+lines",
        marker=dict(
            size=2,
            color='black',
        ),
        line=dict(
            width=1,
            color='yellow',
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=[wp[0, 1]],
        y=[wp[0, 0]],
        z=[-wp[0, 2]],
        mode="markers",
        marker=dict(
            size=20,
            color='red',
        ),
    ))

    plotly.offline.plot(fig, filename='/Users/yaolin/Downloads/test_wp.html', auto_open=True)
    import pandas as pd
    df = pd.DataFrame(wp, columns=['lat', 'lon', 'depth'])
    df.to_csv("/Users/yaolin/Downloads/wp.csv", index=False)




