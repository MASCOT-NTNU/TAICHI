""" Unit test for SPDE helper
This module tests the SPDE helper object.
"""

from unittest import TestCase
from SPDE.SPDEHelper import SPDEHelper
import numpy as np
from numpy import testing


class TestSPDE(TestCase):
    """
    Test class for spde helper class.
    """

    def setUp(self) -> None:
        self.spde_helper = SPDEHelper()
        pass

    def test_get_spde_grid(self):
        self.spde_helper.construct_gmrf_grid()
        pass

    def test_get_ind_from_location(self):
        # c1: one location
        ide = 10
        loc = self.spde_helper.get_location_from_ind(ide)
        id = self.spde_helper.get_ind_from_location(loc)
        self.assertEqual(ide, id)

        # c2: more locations
        ide = [10, 12]
        loc = self.spde_helper.get_location_from_ind(ide)
        id = self.spde_helper.get_ind_from_location(loc)
        self.assertIsNone(testing.assert_array_equal(ide, id))

    def test_get_eibv_at_locations(self):
        id = [1, 2, 3, 4, 5]
        loc = self.spde_helper.get_location_from_ind(id)
        eibv = self.spde_helper.get_eibv_at_locations(loc)

        pass


