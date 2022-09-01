""" Unit test for GMRF
This module tests the GMRF object.
"""

from unittest import TestCase
from GMRF.GMRF import GMRF
import numpy as np
from numpy import testing


class TestGMRF(TestCase):
    """
    Test class for spde helper class.
    """

    def setUp(self) -> None:
        self.gmrf = GMRF()

    def test_get_spde_grid(self) -> None:
        self.gmrf.construct_gmrf_grid()

    def test_get_ind_from_location(self) -> None:
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

    def test_get_ibv(self) -> None:
        """
        Test if GMRF is able to compute IBV
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

    def test_get_eibv_at_locations(self) -> None:
        """
        Test if it can return eibv for the given locations.
        """
        # id = [1, 2, 3, 4, 5]
        id = np.random.randint(0, 1000, 5)
        loc = self.gmrf.get_location_from_ind(id)
        eibv = self.gmrf.get_eibv_at_locations(loc)
        self.assertIsNotNone(eibv)

    def test_assimilate_data(self):
        """
        Test if it can assimilate data with given dataset.
        - 100 grid points within the grid.
        - 10 replicates with 10 grid points not within the grid.
        - no location.
        """
        grid = self.gmrf.get_gmrf_grid()
        # c1: grid points on grid
        ind = np.random.randint(0, grid.shape[0], 100)
        x = grid[ind, 0]
        y = grid[ind, 1]
        z = grid[ind, 2]
        v = np.zeros_like(z)
        dataset = np.stack((x, y, z, v), axis=1)
        ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
        dx = grid[ind_min, 0] - x
        dy = grid[ind_min, 1] - y
        dz = grid[ind_min, 2] - z
        gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        self.assertLess(np.amax(gap), .01)

        # c2: random locations
        for i in range(10):
            ind = np.random.randint(0, grid.shape[0], 10)
            x = grid[ind, 0] + np.random.randn(len(ind))
            y = grid[ind, 1] + np.random.randn(len(ind))
            z = grid[ind, 2] + np.random.randn(len(ind))
            v = np.zeros_like(z)
            dataset = np.stack((x, y, z, v), axis=1)
            ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
            dx = grid[ind_min, 0] - x
            dy = grid[ind_min, 1] - y
            dz = grid[ind_min, 2] - z
            gap = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            self.assertLess(np.amax(gap), 5.2)

        # c3: no location
        dataset = np.empty([0, 4])
        ida, sal_a, ind_min = self.gmrf.assimilate_data(dataset)
        self.assertTrue([True if len(ida) == 0 else False])

