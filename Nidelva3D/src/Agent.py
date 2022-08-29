"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
"""
import numpy as np
import os
from WGS import WGS
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from WaypointGraph import WaypointGraph
from Planner.Myopic3D import Myopic3D
from AUVSimulator.AUVSimulator import AUVSimulator


class Agent:

    __loc_start = np.array([0, 0, 0])
    __loc_end = np.array([0, 0, 0])

    def setup_operational_area(self):
        box = np.load(os.getcwd() + "/GMRF/models/grid.npy")
        polygon = box[:, 2:]
        polygon_xy = np.stack((WGS.latlon2xy(polygon[:, 0], polygon[:, 1])), axis=1)
        polygon_b = sort_polygon_vertices(polygon_xy)

        self.polygon_border = polygon_b
        self.polygon_obstacle = [[[]]]
        self.depths = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        self.neighbour_distance = 120
        self.wp = WaypointGraph()

        self.wp.set_neighbour_distance(self.neighbour_distance)
        self.wp.set_depth_layers(self.depths)
        self.wp.set_polygon_border(self.polygon_border)
        self.wp.set_polygon_obstacles(self.polygon_obstacle)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()

    def setup_agent(self):
        # s1: setup planner
        self.myopic = Myopic3D(self.wp)

        # s2: setup AUV
        self.auv = AUVSimulator()

    def run(self):
        self.auv.move_to_location(self.__loc_start)
        ctd_data = self.auv.get_ctd_data()
        self.myopic.gmrf.assimilate_data(ctd_data)
        wp = self.myopic.get_next_waypoint()
        self.auv.move_to_location(wp)
        i = 0
        NUMS = 60
        while i < NUMS:
            """
            """
            # s1: sense
            #   - get ctd along path from the simulator
            ctd_data = self.auv.get_ctd_data()
            #   - update field
            self.myopic.gmrf.assimilate_data(ctd_data)
            # s2: plan
            #   - plan next waypoint
            wp = self.myopic.get_next_waypoint()
            #   - update planner
            self.myopic.update_planner()
            # s3: act
            #   - move to location
            self.auv.move_to_location(wp)


if __name__ == "__main__":
    a = Agent()


