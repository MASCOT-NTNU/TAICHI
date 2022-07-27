"""
This modules handles the waypointgraph-related problems.
"""
from typing import List, Any

import numpy as np
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from shapely.geometry import Polygon, Point
from Nidelva3D.src.usr_func.is_list_empty import is_list_empty


class WaypointGraph:

    def __init__(self):
        self.__waypoints = np.empty([0, 3])  # put it inside the initialisationm to avoid mutation.
        self.__neighbour = dict()

    def construct_waypoints(self, polygon_border: np.ndarray, polygon_obstacles: list,
                            depths: list, distance_neighbour: float) -> None:
        """

        Args:
            polygon_border: border vertices defined by [[x1, y1], [x2, y2], ..., [xn, yn]].
            polygon_obstacles: multiple obstalce vertices defined by [[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]].
            depths:
            distance_neighbour: distance between neighbouring waypoints.

        Returns:

        """

        xmin, ymin = map(np.amin, [polygon_border[:, 0], polygon_border[:, 1]])
        xmax, ymax = map(np.amax, [polygon_border[:, 0], polygon_border[:, 1]])

        y_gap = distance_neighbour * cos(radians(60)) * 2
        x_gap = distance_neighbour * sin(radians(60))

        gx = np.arange(xmin, xmax, x_gap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(ymin, ymax, y_gap)

        pbs = Polygon(polygon_border)

        def border_contains(p):
            return pbs.contains(p)

        obs_free = True
        if not is_list_empty(polygon_obstacles):
            pos = []
            for po in polygon_obstacles:
                pos.append(Polygon(po))
            obs_free = False

        def obstacles_contain(p):
            obs = False
            for posi in pos:
                if posi.contains(p):
                    obs = True
                    break
            return obs

        d2 = []
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + y_gap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                p = Point(x, y)
                if obs_free:
                    if border_contains(p):
                        d2.append([x, y])
                else:
                    if border_contains(p) and not obstacles_contain(p):
                        d2.append([x, y])

        multiple_depth_layer = False
        if len(depths) > 1:
            multiple_depth_layer = True

        for i in range(len(depths)):
              for j in range(len(d2)):
                self.__waypoints = np.append(self.__waypoints,
                                             np.array([d2[j][0], d2[j][1], depths[i]]).reshape(1, -1), axis=0)
        self.__waypoints = np.array(self.__waypoints)

        gxy = self.__waypoints[:, :2]
        deucli = cdist(self.__waypoints, self.__waypoints, "euclidean")
        if multiple_depth_layer:
            gz = self.__waypoints[:, 2].reshape(-1, 1)
            dg = np.abs(depths[1] - depths[0])
            dellip = (cdist(gxy, gxy, "sqeuclidean") / (1.5 * distance_neighbour)**2 +
                      cdist(gz, gz, "sqeuclidean") / (1.5 * dg)**2)  # TODO: check a more elegant way to replace 1.5
        else:
            dellip = cdist(gxy, gxy, "sqeuclidean") / (1.5 * distance_neighbour) ** 2
        for i in range(len(deucli)):
            nb_ind = np.where((dellip[i] <= 1) * (deucli[i] >= 5))[0]
            self.__neighbour[i] = list(nb_ind)

    def get_legal_hexgonal_points(self, polygon_border: np.ndarray, polygon_obstacles: list,
                                  distance_neighbour: float) -> list:
        """ Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour.
        Sets the boundary and neighbour distance for the discretization under NED coordinate system.
        - N: North
        - E: East
        - D: Down

        Args:
            polygon_border: border vertices defined by [[x1, y1], [x2, y2], ..., [xn, yn]].
            polygon_obstacles: multiple obstalce vertices defined by [[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]].
            distance_neighbour: distance between neighbouring waypoints.


        The resulting grid will be like:
            _________
           /  .   .  \
          /  .  /\   .\
          \   ./__\   .\
           \.   .   .  /
            \_________/

        Returns:
            list of coordinates in xy coordinates:
            [[0, 0],
             [1, 0],
             ...
             [n, n]]

        """

        # return hexgonal2d

    def get_waypoint_from_ind(self, ind: int) -> list[Any]:
        """
        Return waypoint locations using ind

        Returns:

        """
        return self.__waypoints[ind, :]

    def get_ind_from_waypoint(self, waypoint: np.ndarray) -> np.ndarray:
        """

        Args:
            waypoint:

        Returns:

        """

        d = cdist(self.__waypoints, waypoint)
        return np.argmin(d, axis=0)

    def get_ind_neighbours(self, ind: int) -> list:
        """

        Args:
            ind:

        Returns:

        """
        return self.__neighbour[ind]

    def get_waypoints(self) -> np.ndarray:
        """
        Returns:
        """
        return self.__waypoints

    def get_neighbour_hash(self):
        return self.__neighbour


#%%
s = Polygon([[[0, 0],
             [0, 1],
             [1, 1],
             [1, 0]],
             [[2, 2],
              [3, 2],
              [3, 3],
              [2, 3]]])




