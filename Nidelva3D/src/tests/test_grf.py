"""
Test module for the GRF class.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

"""

from unittest import TestCase
from GRF.GRF import GRF
import numpy as np
import matplotlib.pyplot as plt


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.grf = GRF()

    def test_get_gmrf_grid(self):
        pass