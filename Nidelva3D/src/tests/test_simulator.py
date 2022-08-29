""" Unit test for AUV Simulator
"""

from unittest import TestCase
from AUVSimulator.AUVSimulator import AUVSimulator
import numpy as np
from numpy import testing


def value(x, y, z):
    return 2 * x + 3 * y + 4 * z


class TestAUVSimulator(TestCase):

    def setUp(self) -> None:
        self.auv = AUVSimulator()
        xv = np.arange(10)
        yv = np.arange(10)
        zv = np.linspace(0, 1, 5)
        x, y, z = np.meshgrid(xv, yv, zv)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)
        v = value(x, y, z)
        truth = np.hstack((x, y, z, v))
        self.auv.ctd.setup_ctd(truth)

    def test_move_to_location(self):
        """
        Test if the AUV moves according to the given direction.
        """
        # c1: starting location
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), np.array([0, 0, 0])))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0, 0])))

        # c2: move to another location
        loc_new = np.array([10, 10, .5])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0, 0])))

        # c3: move to another location
        loc_new = np.array([20, 20, 1.])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([10, 10, .5])))

    def test_data_collection(self):
        # c1: After it has moved to a location.
        loc_new = np.array([10, 10, .5])
        self.auv.move_to_location(loc_new)
        df = self.auv.get_ctd_data()

        # c2:

