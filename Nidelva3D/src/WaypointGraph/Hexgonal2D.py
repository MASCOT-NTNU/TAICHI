"""
This object handles hexgonal grid generation
"""
from math import cos, sin, radians, degrees
import numpy as np
from ..usr_func import isEven


class Hexgonal2D:

    _x_range = 0
    _y_range = 0
    _distance_neighbour = 0

    def __init__(self):
        pass

    def setup(self, xrange: float, yrange: float, distance_neighbour: float) -> None:
        """ Sets the boundary and neighbour distance for the discretization. NED(North, East, Down) coordinate system is applied.

        Args:
            xrange: distance span across latitude-direction.
            yrange: distance span across longtitude-direction.
            distance_neighbour: distance between neighbouring waypoints.

        """
        self._x_range = xrange
        self._y_range = yrange
        self._distance_neighbour = distance_neighbour

    def get_hexgonal_discretization(self):
        """ Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour

        """
        y_gap = self._distance_neighbour * cos(radians(60)) * 2
        x_gap = self._distance_neighbour * sin(radians(60))

        gx = np.arange(0, self._x_range, x_gap)
        gy = np.arange(0, self._y_range, y_gap)

        hexgonal2d = []
        for i in range(len(gy)):
            for j in range(len(gx)):
                if isEven(j):
                    x = gx[j]
                    y = gy[i] + y_gap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                hexgonal2d.append([x, y])
        hexgonal2d = np.array(hexgonal2d)
        return hexgonal2d

