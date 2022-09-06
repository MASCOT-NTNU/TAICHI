""" Unit test for Visualiser
"""

from unittest import TestCase
from Visualiser.Visualiser import Visualiser


class TestVisualiser(TestCase):

    def setUp(self) -> None:
        self.v = Visualiser()
        pass

    def test_plot(self):
        self.v.plot_agent()
        pass

