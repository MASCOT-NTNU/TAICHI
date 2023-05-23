"""
This module produces a 3D waypoint graph for the path planning for the AUV.

Created on 2023-05-21
:Author: Yaolin Ge
:Email: geyaolin@gmail.com

Coordinate system: NED (North-East-Down)

Input:
    - polygon_border: the polygon defines the border of the area.
        - np.ndarray.
        - border vertices.
        - np.array([[x1, y1],
                    [x2, y2],
                    ...,
                    [xn, yn]]).
        - must not be empty or none.
    - polygon_obstacles: the polygons define the obstacles in the area.
        - list of np.ndarray.
        - multiple obstalce vertices.
        - np.array([[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]]).
        - can be an empty list.
    - depths: depth layers to be explored in the mission.
        - list of float.
        - [d0, d1, d2, ..., dn].
    - distance_neighbour: distance between neighbouring waypoints.
        - float.

Methodology:
- First, it calculates the border limits and the gap distance between neighbouring waypoints.
- Second, it produces a rectangular grid using (xrange, yrange, distance_neighbour) as the parameters.
- Third,
    - it loops each row of the grid and shifts the waypoints in the odd rows by half of the gap distance.
    - it checks if the waypoints are inside the polygon border and outside the polygon obstacles.
- Fourth, it expands the 2D grid to 3D by adding the depth layers, and the lateral distribution is the same.

Output:
- waypoints: an array of waypoints.
    - np.ndarray.
    - np.array([[x0, y0, z0],
                [x1, y1, z1],
                ...
                [xn, yn, zn]]).
- neighbour_hash_table: a hash table that stores the indices of the neighbouring waypoints.
    - dict.
    - {0: [1, 2, 3],
       1: [0, 2, 3],
       ...,
       n: [0, 1, 2]}.
"""
from usr_func.is_list_empty import is_list_empty
from math import cos, sin, radians
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from typing import Any, Union


class WaypointGraph:
    """
    This class produces a 3D waypoint graph for the path planning for the AUV.

    It generates the 3D waypoint graph by stretching the 2D grid to 3D.

    Attributes:
        - waypoints: an array of waypoints.
        - neighbour_hash_table: a hash table that stores the indices of the neighbouring waypoints.
        - neighbour_distance: distance between neighbouring waypoints.
        - depths: depth layers to be explored in the mission.
        - polygon_border: the polygon defines the border of the area.
        - polygon_obstacles: the polygons define the obstacles in the area.

    """

    def __init__(self, neighbour_distance: float = 0, depths: np.ndarray = np.array([0]),
                 polygon_border: np.ndarray = np.array([[0, 0], [0, 0]]),
                 polygon_obstacles: list = None) -> None:
        self.__waypoints = np.empty([0, 3])  # put it inside the initialisation to avoid mutation.
        self.__neighbour_hash_table = {}
        self.__neighbour_distance = neighbour_distance
        self.__depths = depths
        self.__polygon_border = polygon_border
        self.__polygon_obstacles = polygon_obstacles
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__polygon_obstacles_shapely = [Polygon(polygon_obstacle) for polygon_obstacle in self.__polygon_obstacles]

        ''' To be deleted '''
        self.__num_of_depth_layers = len(self.__depths)
        # self.multiple_depth_layer = False
        # self.no_depth_layers = len(self.__depths)
        # if self.no_depth_layers > 1:
        #     self.multiple_depth_layer = True
        ''' To be deleted '''

        if not is_list_empty(self.__polygon_obstacles):
            self.obs_free = False
        else:
            self.obs_free = True
        self.construct_waypoints()
        self.construct_hash_neighbours()

    def construct_waypoints(self) -> None:
        """ Construct the waypoints for the 3D waypoint graph given the border and obstacles.

        Methods:
            - First, it calculates the x, y limits and gaps using __cal_xy_limits_and_gaps().
            - Second, it produces a rectangular grid using (xrange, yrange, neighbour_distance) as the parameters.
            - Third, it loops each row of the grid and shifts the waypoints in the odd rows by half of the gap distance.
            - Fourth, it checks if the waypoints are inside the polygon border and outside the polygon obstacles.
            - Fifth, it expands the 2D grid to 3D by adding the depth layers, and the lateral distribution is the same.
        """
        # s1, get x, y ranges and gaps.
        self.__cal_xy_limits_and_gaps()

        # s2, get 2D legal grid points.
        xv = np.arange(self.__xmin, self.__xmax, self.__xgap)
        yv = np.arange(self.__ymin, self.__ymax, self.__ygap)
        waypoint_2d = []
        counter_waypoint_2d = 0
        for i in range(len(yv)):
            for j in range(len(xv)):
                if j % 2 == 0:
                    x = xv[j]
                    y = yv[i] + self.__ygap / 2
                else:
                    x = xv[j]
                    y = yv[i]
                candidate_waypoint_2d = Point(x, y)
                if self.obs_free:
                    if self.__border_contains(candidate_waypoint_2d):
                        waypoint_2d.append([x, y])
                        counter_waypoint_2d += 1
                else:
                    if self.__border_contains(candidate_waypoint_2d) and \
                            not self.__obstacles_contain(candidate_waypoint_2d):
                        waypoint_2d.append([x, y])
                        counter_waypoint_2d += 1
        for i in range(self.__num_of_depth_layers):
            for j in range(counter_waypoint_2d):
                self.__waypoints = np.append(self.__waypoints,
                                             np.array([waypoint_2d[j][0], waypoint_2d[j][1],
                                                       self.__depths[i]]).reshape(1, -1), axis=0)
        self.__waypoints = np.array(self.__waypoints)

    def __cal_xy_limits_and_gaps(self) -> None:
        """
        Calculate the border xy limits and the gap distance between neighbouring waypoints.
        """
        x = self.__polygon_border[:, 0]
        y = self.__polygon_border[:, 1]
        self.__xmin, self.__ymin = map(np.amin, [x, y])
        self.__xmax, self.__ymax = map(np.amax, [x, y])
        self.__ygap = self.__neighbour_distance * cos(radians(60)) * 2
        self.__xgap = self.__neighbour_distance * sin(radians(60))

    def __border_contains(self, point: Point) -> bool:
        """ Test if a point is within the border polygon """
        return self.__polygon_border_shapely.contains(point)

    def __obstacles_contain(self, point: Point) -> bool:
        """ Test if a point is colliding with any obstacle polygons """
        colliding = False
        for polygon_obstacle_shapely in self.__polygon_obstacles_shapely:
            if polygon_obstacle_shapely.contains(point):
                colliding = True
                break
        return colliding

    def construct_hash_neighbours(self) -> None:
        """ Construct the hash table for containing neighbour indices around each waypoint.

        Methods:
            - Loop through each waypoint.
            - First, get the indices for the adjacent depth layers.
            - Second, loop through each adjacent depth layer.
                - Get the indices for the lateral neighbours in that layer.
            - Last, append all the neighbour indices for each waypoint.
        """
        self.num_of_waypoint = self.__waypoints.shape[0]
        ERROR_BUFFER = 1
        for i in range(self.num_of_waypoint):
            # s0, separate xy and depth
            current_xy = self.__waypoints[i, 0:2].reshape(1, -1)
            depth = self.__waypoints[i, 2]
            ind_depth = np.where(self.__depths == depth)[0][0]

            # s1, get the indices for the adjacent depth layers.
            ind_upper_depth_layer = ind_depth + 1 if ind_depth < self.__num_of_depth_layers - 1 else ind_depth
            ind_lower_depth_layer = ind_depth - 1 if ind_depth > 0 else 0
            indices_of_all_depth_layers = np.unique([ind_depth, ind_lower_depth_layer, ind_upper_depth_layer])
            all_depth_layers = self.__depths[indices_of_all_depth_layers]

            # s2, loop through each adjacent depth layer and get the indices for the lateral neighbours in that layer.
            ind_neighbours = []
            for depth_layer in all_depth_layers:
                ind_candidates_in_this_depth_layer = np.where(self.__waypoints[:, 2] == depth_layer)[0]
                candidate_xy = self.__waypoints[ind_candidates_in_this_depth_layer, 0:2]
                distance = cdist(candidate_xy, current_xy)
                ind_candidate_neighbours_local = np.where((distance <= self.__neighbour_distance + ERROR_BUFFER) *
                                                    (distance >= self.__neighbour_distance - ERROR_BUFFER))[0]
                for ind_candidate_neighbour_local in ind_candidate_neighbours_local:
                    ind_neighbours.append(ind_candidates_in_this_depth_layer[ind_candidate_neighbour_local])  # convert to global index
            self.__neighbour_hash_table[i] = ind_neighbours

    def get_waypoints(self) -> np.ndarray:
        """
        Returns: waypoints
        """
        return self.__waypoints

    def get_neighbour_hash_table(self) -> dict:
        """
        Returns: neighbour hash table
        """
        return self.__neighbour_hash_table

    def get_polygon_border(self) -> np.ndarray:
        """
        Returns: border polygon
        """
        return self.__polygon_border

    def get_polygon_obstacles(self) -> list:
        """
        Returns: obstacles' polygons.
        """
        return self.__polygon_obstacles

    def get_waypoint_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return self.__waypoints[ind, :]

    def get_ind_from_waypoint(self, waypoint: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Return the index of the closest waypoint to the given waypoint.

        Method:
            - First, it calculates the distance between the given waypoint and all the waypoints.
            - Second, it returns the index of the waypoint with the smallest distance.

        Input:
            waypoint: np.array([xp, yp, zp])

        Returns:
            ind: index of the closest waypoint.
                - integer if the given waypoint is a single waypoint.
                - None if the given waypoint is empty.
                - np.ndarray of indices if the given waypoint is an array of waypoints.
        """
        if len(waypoint) > 0:
            dimension = waypoint.ndim
            if dimension == 1:
                distance = cdist(self.__waypoints, waypoint.reshape(1, -1))
                return np.argmin(distance, axis=0)[0]
            elif dimension == 2:
                distance = cdist(self.__waypoints, waypoint)
                return np.argmin(distance, axis=0)
            else:
                return None
        else:
            return None

    def get_ind_neighbours(self, ind: int) -> list:
        """
        Args:
            ind: waypoint index
        Returns: neighbour indices
        """
        return self.__neighbour_hash_table[ind]

    @staticmethod
    def get_vector_between_two_waypoints(wp1: np.ndarray, wp2: np.ndarray) -> np.ndarray:
        """ Get a vector from wp1 to wp2.

        Args:
            wp1: np.array([x1, y1, z1])
            wp2: np.array([x2, y2, z2])

        Returns:
            vec: np.array([[x2 - x1],
                           [y2 - y1],
                           [z2 - z1]])

        """
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        dz = wp2[2] - wp1[2]
        vec = np.vstack((dx, dy, dz))
        return vec

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def set_depth_layers(self, value: list) -> None:
        """ Set the depth layers """
        self.__depths = np.array(value)

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        self.__polygon_border = value
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

    def set_polygon_obstacles(self, value: list) -> None:
        """ Set the polygons for obstacles, can have multiple obstacles """
        self.__polygon_obstacles = value
        self.obs_free = True
        if not is_list_empty(self.__polygon_obstacles):
            self.__polygon_obstacles_shapely = []
            for polygon in self.__polygon_obstacles:
                self.__polygon_obstacles_shapely.append(Polygon(polygon))
            self.obs_free = False


if __name__ == "__main__":
    w = WaypointGraph(neighbour_distance=10, depths=np.array([0, 1]),
                      polygon_border=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
                      polygon_obstacles=[np.array([[10, 10], [20, 10], [20, 20], [10, 20]]),
                                         np.array([[30, 30], [40, 30], [40, 40], [30, 40]])])
    print(w.get_waypoints())