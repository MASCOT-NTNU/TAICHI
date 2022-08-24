"""
Myopic3D path planner determines the next waypoint according to minimum EIBV criterion.
"""
from Planner.Planner import Planner
from WaypointGraph import WaypointGraph
from SPDE.GMRF import GMRF
from usr_func.is_list_empty import is_list_empty
import numpy as np


class Myopic3D(Planner):
    """
    Myopic3D planner.
    """

    def __init__(self, wp: WaypointGraph) -> None:
        super().__init__()
        self.wp = wp
        self.gmrf = GMRF()

    def get_candidates(self):
        """
        Filter sharp turn, bottom up and dive down behaviours.

        It computes the dot product between two vectors. One is from the previous waypoint to the current waypoint.
        The other is from the current waypoint to the potential next waypoint.
        Example:
            >>> wp_prev = np.array([x1, y1, z1])
            >>> wp_curr = np.array([x2, y2, z2])
            >>> vec1 = wp_curr - wp_prev
            >>> for wp in wp_neighbours:
            >>>     wp = np.array([xi, yi, zi])
            >>>     vec2 = wp - wp_curr
            >>>     if dot(vec1.T, vec2) >= 0:
            >>>         append(wp)

        Returns:
            id_smooth: filtered candidate indices in the waypointgraph.
            id_neighbours: all the neighbours at the current location.
        """
        id_prev = self.get_previous_index()
        id_curr = self.get_current_index()

        # s1: get vec from previous wp to current wp.
        wp_prev = self.wp.get_waypoint_from_ind(id_prev)
        wp_curr = self.wp.get_waypoint_from_ind(id_curr)
        vec1 = self.wp.get_vector_between_two_waypoints(wp_prev, wp_curr)

        # s2: get all neighbours.
        id_neighbours = self.wp.get_ind_neighbours(id_curr)

        # s3: smooth neighbour locations.
        id_smooth = []
        for iid in id_neighbours:
            wp_n = self.wp.get_waypoint_from_ind(iid)
            vec2 = self.wp.get_vector_between_two_waypoints(wp_curr, wp_n)
            if vec1.T @ vec2 >= 0:
                id_smooth.append(iid)
        return id_smooth, id_neighbours

    def get_next_waypoint(self):
        """
        Get next waypoint according to minimum EIBV criterion.

        If no possible candiate locations were found. Then a random location in the neighbourhood is selected.
        Also the pioneer index can be modified here.

        Returns:
            wp_next: designed next waypoint location.
        """
        id_smooth, id_neighbours = self.get_candidates()
        if not is_list_empty(id_smooth):
            # s1: get candidate locations
            locs = self.wp.get_waypoint_from_ind(id_smooth)
            # s2: get eibv at that waypoint
            eibv = self.gmrf.get_eibv_at_locations(locs)
            wp_next = locs[np.argmin(eibv)]
        else:
            rng_ind = np.random.randint(0, len(id_neighbours))
            id_next = id_neighbours[rng_ind]
            self.set_next_index(id_next)
            wp_next = self.wp.get_waypoint_from_ind(id_next)
        return wp_next


