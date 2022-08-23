"""
Myopic3D path planner determines the next waypoint according to minimum EIBV criterion.
"""
from Planner.Planner import Planner
from WaypointGraph import WaypointGraph
from SPDE.SPDEHelper import SPDEHelper
from usr_func.is_list_empty import is_list_empty
import numpy as np


class Myopic3D(Planner):

    def __init__(self, wp: WaypointGraph) -> None:
        super().__init__()
        self.wp = wp
        self.spde_helper = SPDEHelper()

    def get_candidates(self):
        wp_prev = self.wp.get_waypoint_from_ind(self._id_prev)
        wp_now = self.wp.get_waypoint_from_ind(self._id_now)
        vec1 = self.wp.get_vector_between_two_waypoints(wp_prev, wp_now)
        id_neighbours = self.wp.get_ind_neighbours(self._id_now)
        id_smooth = []
        for iid in id_neighbours:
            wp_n = self.wp.get_waypoint_from_ind(iid)
            vec2 = self.wp.get_vector_between_two_waypoints(wp_n, wp_now)
            if vec1.T @ vec2 >= 0:
                id_smooth.append(iid)
        return id_smooth, id_neighbours

    def get_next_waypoint(self):
        id_smooth, id_neighbours = self.get_candidates()
        if not is_list_empty(id_smooth):
            # s1: get candidate locations
            locs = self.wp.get_waypoint_from_ind(id_smooth)
            # s2: get eibv at that waypoint
            eibv = self.spde_helper.get_eibv_at_locations(locs)
            wp_next = locs[np.argmin(eibv)]
        else:
            rng_ind = np.random.randint(0, len(id_neighbours))
            id_next = id_neighbours[rng_ind]
            self.set_next_index(id_next)
            wp_next = self.wp.get_waypoint_from_ind(id_next)
        return wp_next


