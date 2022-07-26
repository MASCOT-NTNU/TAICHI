"""
This object handles hexgonal grid generation
"""
from math import cos, sin, radians
import numpy as np


class Hexgonal2D:

    _x_range = 0
    _y_range = 0
    _distance_neighbour = 0

    def __init__(self) -> None:
        """ Generates the object to handle hexgonal 2d discretization.

        """

    def setup(self, xrange: float, yrange: float, distance_neighbour: float) -> None:
        """ Sets the boundary and neighbour distance for the discretization under NED coordinate system.
        - N: North
        - E: East
        - D: Down

        Args:
            xrange: distance span across latitude-direction.
            yrange: distance span across longtitude-direction.
            distance_neighbour: distance between neighbouring waypoints.

        """
        self._x_range = xrange
        self._y_range = yrange
        self._distance_neighbour = distance_neighbour

    def get_hexgonal_discretization(self):
        """ Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour.

        The resulting grid will be like:
        .   .   .   .   .
          .   .   .   .   .
        .   .   .   .   .
          .   .   .   .   .
        .   .   .   .   .
          .   .   .   .   .
        """
        y_gap = self._distance_neighbour * cos(radians(60)) * 2
        x_gap = self._distance_neighbour * sin(radians(60))

        gx = np.arange(0, self._x_range, x_gap)
        gy = np.arange(0, self._y_range, y_gap)

        hexgonal2d = []
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + y_gap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                hexgonal2d.append([x, y])
        return hexgonal2d

