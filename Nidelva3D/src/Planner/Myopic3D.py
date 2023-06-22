"""
Myopic 3D path planner determines the pioneer waypoint according to minimum EIBV criterion.

Author: Youlin Ge
Email: geyaolin@gmail.com
Date: 2023-05-23

Methodology:
    1. Get the current location of the AUV.
    2. Filter the illegal next waypoints.
    3. Compute the EIBV of each candidate waypoint.
    4. Choose the waypoint with the minimum EIBV as the next waypoint.
    5. Update the pioneer waypoint.
"""
from Planner.Planner import Planner
from WaypointGraph import WaypointGraph
from GMRF.GMRF import GMRF
from GRF.GRF import GRF
from WGS import WGS
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from usr_func.is_list_empty import is_list_empty
import numpy as np
import pandas as pd
import os
from typing import Union


class Myopic3D(Planner):
    """
    Myopic3D planner determines the next waypoint according to minimum EIBV criterion.
    """
    # __BOX = np.load(os.getcwd() + "/GMRF/models/grid.npy")
    # __POLYGON = __BOX[:, 2:]
    # __POLYGON_XY = np.stack((WGS.latlon2xy(__POLYGON[:, 0], __POLYGON[:, 1])), axis=1)
    # __POLYGON_BORDER = sort_polygon_vertices(__POLYGON_XY)

    __POLYGON_BORDER_WGS = pd.read_csv(os.getcwd() + "/polygon_border.csv").to_numpy()
    x, y = WGS.latlon2xy(__POLYGON_BORDER_WGS[:, 0], __POLYGON_BORDER_WGS[:, 1])
    __POLYGON_BORDER = np.stack((x, y), axis=1)

    __POLYGON_OBSTACLE = [[[]]]
    __DEPTHS = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    __NEIGHBOUR_DISTANCE = 240

    def __init__(self, kernel: str = "GMRF") -> None:
        super().__init__()
        self.waypoint_graph = WaypointGraph(neighbour_distance=self.__NEIGHBOUR_DISTANCE,
                                            depths=self.__DEPTHS,
                                            polygon_border=self.__POLYGON_BORDER,
                                            polygon_obstacles=self.__POLYGON_OBSTACLE)
        if kernel == "GMRF":
            print("KERNEL: GMRF")
            self.kernel = GMRF()
        elif kernel == "GRF":
            print("KERNEL: GRF")
            self.kernel = GRF()
        else:
            raise ValueError("Kernel must be either GMRF or GRF.")

    def get_candidates_indices(self) -> tuple:
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
        wp_curr = self.waypoint_graph.get_waypoint_from_ind(id_curr)
        wp_next = self.waypoint_graph.get_waypoint_from_ind(id_next)
        vec1 = self.waypoint_graph.get_vector_between_two_waypoints(wp_curr, wp_next)

        # s2: get all neighbours.
        id_neighbours = self.waypoint_graph.get_ind_neighbours(id_next)

        # s3: get visited locations.
        id_visited = self.get_trajectory_indices()

        # s3: smooth neighbour locations.
        id_smooth = []
        for iid in id_neighbours:
            wp_n = self.waypoint_graph.get_waypoint_from_ind(iid)
            vec2 = self.waypoint_graph.get_vector_between_two_waypoints(wp_next, wp_n)
            if vec1.T @ vec2 >= 0 and iid not in id_visited:
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
            locs = self.waypoint_graph.get_waypoint_from_ind(id_smooth)
            # s2: get eibv at that waypoint
            eibv = self.kernel.get_eibv_at_locations(locs)
            print("EIBV: ", eibv)
            id_pioneer = id_smooth[np.argmin(eibv)]
            # wp_pioneer = locs[np.argmin(eibv)]
        else:
            rng_ind = np.random.randint(0, len(id_neighbours))
            id_pioneer = id_neighbours[rng_ind]
            # wp_pioneer = self.wp.get_waypoint_from_ind(id_pioneer)
        self.set_pioneer_index(id_pioneer)
        return id_pioneer

