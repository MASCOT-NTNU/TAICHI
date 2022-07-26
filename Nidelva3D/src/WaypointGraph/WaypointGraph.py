"""
This object handles the waypointgraph-related problems. It employs Hexgonal2D as the building block.
"""

from Hexgonal2D import Hexgonal2D
import numpy as np
from scipy.spatial.distance import cdist


class WaypointGraph(Hexgonal2D):
    _depth_layer = []
    _waypoints = []
    _neighbour = dict()

    def __init__(self):
        pass

    def set_depth_layers(self, depths: list) -> None:
        """ Sets the depth layer as a list

        Args: 


        """
        self._depth_layer = depths
        self._d_depth_gap = np.abs(self._depth_layer[1] - self._depth_layer[0])

    def construct_waypoints(self, xrange: float, yrange: float, distance_neighbour: float) -> None:
        h2d = Hexgonal2D()
        h2d.setup(xrange, yrange, distance_neighbour)
        g2d = h2d.get_hexgonal_discretization()
        # print(g2d)

        for i in range(len(self._depth_layer)):
              for j in range(len(g2d)):
                self._waypoints.append([g2d[j][0], g2d[j][1], self._depth_layer[i]])
        
#        print(self._waypoints)
        g = np.array(self._waypoints)
        gxy = g[:, :2]
        gz = g[:, 2].reshape(-1, 1)
        deucli = cdist(g, g, "euclidean")

        dellip = np.sqrt(cdist(gxy, gxy, "sqeuclidean") / (1.5 * self._distance_neighbour)**2 + cdist(gz, gz, "sqeuclidean") / (1.5 * self._d_depth_gap)**2)
        print(self._distance_neighbour)
        print(self._d_depth_gap)
 
    def get_waypoint_from_ind(self):
        pass

    def get_ind_from_waypoint(self):
        pass

    def get_ind_neighbours(self):
        pass


if __name__ == "__main__":
    xrange = 100
    yrange = 100
    distance_neighbour = 12
    w = WaypointGraph()
    w.set_depth_layers([0, 0.5, 1.0])
    w.construct_waypoints(xrange, yrange, distance_neighbour)
