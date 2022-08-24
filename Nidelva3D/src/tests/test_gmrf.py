""" Unit test for SPDE helper
This module tests the SPDE helper object.
"""

from unittest import TestCase
from SPDE.GMRF import GMRF
import numpy as np
from numpy import testing


class TestGMRF(TestCase):
    """
    Test class for spde helper class.
    """

    def setUp(self) -> None:
        self.gmrf = GMRF()

    def test_get_spde_grid(self):
        self.gmrf.construct_gmrf_grid()

    def test_get_ind_from_location(self):
        """
        Test if given a location, it will return the correct index in GMRF grid.
        """
        # c1: one location
        ide = 10
        loc = self.gmrf.get_location_from_ind(ide)
        id = self.gmrf.get_ind_from_location(loc)
        self.assertEqual(ide, id)

        # c2: more locations
        ide = [10, 12]
        loc = self.gmrf.get_location_from_ind(ide)
        id = self.gmrf.get_ind_from_location(loc)
        self.assertIsNone(testing.assert_array_equal(ide, id))

    def test_get_ibv(self):
        """

        """
        # c1: mean at threshold
        threshold = 0
        mu = np.array([0])
        sigma_diag = np.array([1])
        ibv = self.gmrf.get_ibv(threshold, mu, sigma_diag)
        self.assertEqual(ibv, .25)

        # c2: mean further away from threshold
        threshold = 0
        mu = np.array([3])
        sigma_diag = np.array([1])
        ibv = self.gmrf.get_ibv(threshold, mu, sigma_diag)
        self.assertLess(ibv, .01)

    def test_get_eibv_at_locations(self):
        # id = [1, 2, 3, 4, 5]
        id = np.random.randint(0, 1000, 5)
        loc = self.gmrf.get_location_from_ind(id)
        eibv = self.gmrf.get_eibv_at_locations(loc)
        self.assertIsNotNone(eibv)
