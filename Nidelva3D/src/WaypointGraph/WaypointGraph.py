"""
Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour.
Sets the boundary and neighbour distance for the discretization under NED coordinate system.
- N: North
- E: East
- D: Down

Args:
    polygon_border: border vertices defined by [[x1, y1], [x2, y2], ..., [xn, yn]].
    polygon_obstacles: multiple obstalce vertices defined by [[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]].
    depths: multiple depth layers [d0, d1, d2, ..., dn].
    distance_neighbour: distance between neighbouring waypoints.

The resulting grid will be like:
    _________
   /  .   .  \
  /  .  /\   .\
  \   ./__\   .\
   \.   .   .  /
    \_________/

Get:
    Waypoints: [[x0, y0, z0],
               [x1, y1, z1],
               ...
               [xn, yn, zn]]
    Neighbour hash tables: {0: [1, 2, 3], 1: [0, 2, 3], ..., }
"""
from typing import Any
import numpy as np
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from shapely.geometry import Polygon, Point
from Nidelva3D.src.usr_func.is_list_empty import is_list_empty


class WaypointGraph:

    def __init__(self):
        self.__waypoints = np.empty([0, 3])  # put it inside the initialisation to avoid mutation.
        self.__neighbour = dict()
        self.__neighbour_distance = 0
        self.__depths = []
        self.__polygon_border = np.array([[0, 0],
                                          [0, 0],
                                          [0, 0]])
        self.__polygon_obstacles = [[[]]]

    def set_neighbour_distance(self, value: float) -> None:
        self.__neighbour_distance = value

    def set_depth_layers(self, value: list) -> None:
        self.__depths = np.array(value)

    def set_polygon_border(self, value: np.ndarray) -> None:
        self.__polygon_border = value
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

    def set_polygon_obstacles(self, value: list) -> None:
        self.__polygon_obstacles = value
        self.obs_free = True
        if not is_list_empty(self.__polygon_obstacles):
            self.__polygon_obstacles_shapely = []
            for po in self.__polygon_obstacles:
                self.__polygon_obstacles_shapely.append(Polygon(po))
            self.obs_free = False

    def get_xy_limits(self):
        xb = self.__polygon_border[:, 0]
        yb = self.__polygon_border[:, 1]
        self.xmin, self.ymin = map(np.amin, [xb, yb])
        self.xmax, self.ymax = map(np.amax, [xb, yb])

    def get_xy_gaps(self):
        self.ygap = self.__neighbour_distance * cos(radians(60)) * 2
        self.xgap = self.__neighbour_distance * sin(radians(60))

    def border_contains(self, point):
        return self.__polygon_border_shapely.contains(point)

    def obstacles_contain(self, point):
        obs = False
        for posi in self.__polygon_obstacles_shapely:
            if posi.contains(point):
                obs = True
                break
        return obs

    def construct_waypoints(self) -> None:
        self.get_xy_limits()
        self.get_xy_gaps()

        gx = np.arange(self.xmin, self.xmax, self.xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(self.ymin, self.ymax, self.ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + self.ygap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                p = Point(x, y)
                if self.obs_free:
                    if self.border_contains(p):
                        grid2d.append([x, y])
                        counter_grid2d += 1
                else:
                    if self.border_contains(p) and not self.obstacles_contain(p):
                        grid2d.append([x, y])
                        counter_grid2d += 1

        self.multiple_depth_layer = False
        self.no_depth_layers = len(self.__depths)
        if self.no_depth_layers > 1:
            self.multiple_depth_layer = True

        for i in range(self.no_depth_layers):
              for j in range(counter_grid2d):
                self.__waypoints = np.append(self.__waypoints,
                                             np.array([grid2d[j][0], grid2d[j][1],
                                                       self.__depths[i]]).reshape(1, -1), axis=0)
        self.__waypoints = np.array(self.__waypoints)

    def construct_hash_neighbours(self):
        # check adjacent depth layers to determine the neighbouring waypoints.
        self.no_waypoint = self.__waypoints.shape[0]
        ERROR_BUFFER = 1
        for i in range(self.no_waypoint):

            # determine ind depth layer
            xy_c = self.__waypoints[i, 0:2].reshape(1, -1)
            d_c = self.__waypoints[i, 2]
            ind_d = np.where(self.__depths == d_c)[0][0]

            # determine ind adjacent layers
            ind_u = ind_d + 1 if ind_d < self.no_depth_layers - 1 else ind_d
            ind_l = ind_d - 1 if ind_d > 0 else 0

            # compute lateral distance
            id = np.unique([ind_d, ind_l, ind_u])
            ds = self.__depths[id]

            ind_n = []
            for ids in ds:
                ind_id = np.where(self.__waypoints[:, 2] == ids)[0]
                xy = self.__waypoints[ind_id, 0:2]
                dist = cdist(xy, xy_c)
                ind_n_temp = np.where((dist <= self.__neighbour_distance + ERROR_BUFFER) *
                                      (dist >= self.__neighbour_distance - ERROR_BUFFER))[0]
                for idt in ind_n_temp:
                    ind_n.append(ind_id[idt])
            self.__neighbour[i] = ind_n

    # def get_hash_neighbours(self):
    #     gxy = self.__waypoints[:, :2]
    #     deucli = cdist(self.__waypoints, self.__waypoints, "euclidean")
    #     if self.multiple_depth_layer:
    #         gz = self.__waypoints[:, 2].reshape(-1, 1)
    #         dg = np.abs(self.__depths[1] - self.__depths[0])
    #         dellip = (cdist(gxy, gxy, "sqeuclidean") / (1.5 * self.__neighbour_distance)**2 +
    #                   cdist(gz, gz, "sqeuclidean") / (1.5 * dg)**2)  # TODO: check a more elegant way to replace 1.5
    #     else:
    #         dellip = cdist(gxy, gxy, "sqeuclidean") / (1.5 * self.__neighbour_distance) ** 2
    #     for i in range(len(deucli)):
    #         nb_ind = np.where((dellip[i] <= 1) * (deucli[i] >= 5))[0]
    #         self.__neighbour[i] = list(nb_ind)

    def get_waypoints(self):
        """
        Returns: waypoints
        """
        return self.__waypoints

    def get_hash_neighbour(self):
        """
        Returns: neighbour hash table
        """
        return self.__neighbour

    def get_polygon_border(self):
        """
        Returns: border polygon
        """
        return self.__polygon_border

    def get_polygon_obstacles(self):
        """
        Returns: obstacles' polygons
        """
        return self.__polygon_obstacles

    def get_waypoint_from_ind(self, ind: int) -> list[Any]:
        """
        Return waypoint locations using ind
        """
        return self.__waypoints[ind, :]

    def get_ind_from_waypoint(self, waypoint: np.ndarray) -> np.ndarray:
        """
        Args:
            waypoint: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """
        d = cdist(self.__waypoints, waypoint)
        return np.argmin(d, axis=0)

    def get_ind_neighbours(self, ind: int) -> list:
        """
        Args:
            ind: waypoint index
        Returns: neighbour indices
        """
        return self.__neighbour[ind]




