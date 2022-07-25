""" Unit test for Hexgonal2D

This module tests the planner object.

"""

import unittest
from src.WaypointGraph.Hexgonal2D import Hexgonal2D
from math import radians, sin, cos
import matplotlib.pyplot as plt
import numpy as np


class TestHexgonal2D(unittest.TestCase):
    """ Test class for planner.

    """

    def test_hexgonal2D_discretization(self):
        """ Tests update planner

        """
        h = Hexgonal2D()
        xr = 1000
        yr = 1000
        d = 120
        h.setup(xr, yr, d)
        actual = h.get_hexgonal_discretization()
        plt.plot(actual[:, 1], actual[:, 0], 'k.')
        plt.show()
        # expected = (d * sin(radians(60)), d * cos(radians(60)) * 2)
        # self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main(exit=False)


