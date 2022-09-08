"""
Myopic3D path planner determines the pioneer waypoint according to minimum EIBV criterion.
It utilises four waypoints to make an asychronous planning system.

- Current waypoint: contains the current location, used to filter illegal next waypoints.
- Next waypoint: contains the next waypoint, and the AUV should go to next waypoint once it arrives at the current one.
- Pioneer waypoint: contains the pioneer waypoint which is one step ahead of the next waypoint.

"""
from Planner.Planner import Planner
from WaypointGraph import WaypointGraph
from GMRF.GMRF import GMRF
from WGS import WGS
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from usr_func.is_list_empty import is_list_empty
import numpy as np
import os
from typing import Union


class Myopic3D(Planner):
    """
    Myopic3D planner determines the next waypoint according to minimum EIBV criterion.
    """
    __BOX = np.load(os.getcwd() + "/GMRF/models/grid.npy")
    __POLYGON = __BOX[:, 2:]
    __POLYGON_XY = np.stack((WGS.latlon2xy(__POLYGON[:, 0], __POLYGON[:, 1])), axis=1)
    __POLYGON_BORDER = sort_polygon_vertices(__POLYGON_XY)
    __POLYGON_OBSTACLE = [[[]]]
    __DEPTHS = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    __NEIGHBOUR_DISTANCE = 120

    def __init__(self) -> None:
        super().__init__()
        self.wp = WaypointGraph()
        self.setup_waypoint_graph()
        self.gmrf = GMRF()

    def setup_waypoint_graph(self) -> None:
        """
        Set the waypoint graph for the whole field according to those polygon constrains.
        """
        self.wp.set_neighbour_distance(self.__NEIGHBOUR_DISTANCE)
        self.wp.set_depth_layers(self.__DEPTHS)
        self.wp.set_polygon_border(self.__POLYGON_BORDER)
        self.wp.set_polygon_obstacles(self.__POLYGON_OBSTACLE)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()

    def get_candidates_indices(self):
        """
        Filter sharp turn, bottom up and dive down behaviours.

        It computes the dot product between two vectors. One is from the previous waypoint to the current waypoint.
        The other is from the current waypoint to the potential next waypoint.
        Example:
            >>> wp_curr = np.array([x1, y1, z1])
            >>> wp_next = np.array([x2, y2, z2])
            >>> vec1 = wp_next - wp_curr
            >>> for wp in wp_neighbours:
            >>>     wp = np.array([xi, yi, zi])
            >>>     vec2 = wp - wp_next
            >>>     if dot(vec1.T, vec2) >= 0:
            >>>         append(wp)

        Returns:
            id_smooth: filtered candidate indices in the waypointgraph.
            id_neighbours: all the neighbours at the current location.
        """
        id_curr = self.get_current_index()
        id_next = self.get_next_index()

        # s1: get vec from previous wp to current wp.
        wp_curr = self.wp.get_waypoint_from_ind(id_curr)
        wp_next = self.wp.get_waypoint_from_ind(id_next)
        vec1 = self.wp.get_vector_between_two_waypoints(wp_curr, wp_next)

        # s2: get all neighbours.
        id_neighbours = self.wp.get_ind_neighbours(id_next)

        # s3: smooth neighbour locations.
        id_smooth = []
        for iid in id_neighbours:
            wp_n = self.wp.get_waypoint_from_ind(iid)
            vec2 = self.wp.get_vector_between_two_waypoints(wp_next, wp_n)
            if vec1.T @ vec2 >= 0:
                id_smooth.append(iid)
        return id_smooth, id_neighbours

    def get_pioneer_waypoint_index(self) -> Union[None, int]:
        """
        Get pioneer waypoint index according to minimum EIBV criterion, which is only an integer.

        If no possible candidate locations were found. Then a random location in the neighbourhood is selected.
        Also the pioneer index can be modified here.

        Returns:
            id_pioneer: designed pioneer waypoint index.
        """
        id_smooth, id_neighbours = self.get_candidates_indices()
        if not is_list_empty(id_smooth):
            # s1: get candidate locations
            locs = self.wp.get_waypoint_from_ind(id_smooth)
            # s2: get eibv at that waypoint
            eibv = self.gmrf.get_eibv_at_locations(locs)
            id_pioneer = id_smooth[np.argmin(eibv)]
            # wp_pioneer = locs[np.argmin(eibv)]
        else:
            rng_ind = np.random.randint(0, len(id_neighbours))
            id_pioneer = id_neighbours[rng_ind]
            # wp_pioneer = self.wp.get_waypoint_from_ind(id_pioneer)
        self.set_pioneer_index(id_pioneer)
        return id_pioneer

