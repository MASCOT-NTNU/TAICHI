"""
Agent object.
"""
import numpy as np
import os
from WGS import WGS
from usr_func.sort_polygon_vertices import sort_polygon_vertices
from WaypointGraph import WaypointGraph
from SPDE.GMRF import GMRF
from Planner.Myopic3D import Myopic3D


class Agent:

    def __init__(self):
        pass

    def setup_operational_area(self):
        box = np.load(os.getcwd() + "/SPDE/models/grid.npy")
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
        # self.waypoints = self.wp.get_waypoints()
        # self.hash_neighbours = self.wp.get_hash_neighbour()

    def setup_sensor(self):
        self.gmrf = GMRF()

    def setup_planner(self):
        self.myopic = Myopic3D(self.wp)

    def setup_actuator(self):

        pass

    def run(self):
        while True:
            """
            """
            # s1: sense
            # s2: plan
            # s3: act
            break
        pass




if __name__ == "__main__":
    a = Agent()


