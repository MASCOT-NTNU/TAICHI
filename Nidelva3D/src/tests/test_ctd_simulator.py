""" Unit test for CTD simulator
"""

from unittest import TestCase
from AUVSimulator.CTDSimulator import CTDSimulator
import numpy as np


def value(x, y, z):
    return 2 * x + 3 * y + 4 * z


class TestCTDSimulator(TestCase):

    def setUp(self) -> None:
        self.ctd = CTDSimulator()
        xv = np.arange(10)
        yv = np.arange(10)
        zv = np.linspace(0, 1, 5)
        x, y, z = np.meshgrid(xv, yv, zv)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        v = value(x, y, z)
        truth = np.hstack((x, y, z, v))
        self.ctd.setup_ctd(truth)

    def test_get_salinity_at_loc(self):
        print("Hello")
        # c1: value at the corner
        x = 0
        y = 0
        z = 0
        loc = np.array([x, y, z])
        v = self.ctd.get_salinity_at_loc(loc)
        self.assertEqual(v, value(x, y, z))

        # c2: value within the field
        x, y, z = 5, 5, .5
        loc = np.array([x, y, z])
        v = self.ctd.get_salinity_at_loc(loc)
        self.assertEqual(v, value(x, y, z))

        # c3: value outside
        x, y, z = 5, 5, 1.5
        loc = np.array([x, y, z])
        v = self.ctd.get_salinity_at_loc(loc)
        self.assertEqual(v, value(x, y, 1))

