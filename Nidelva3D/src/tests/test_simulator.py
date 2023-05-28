"""
Test the simulator module.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28
"""
from unittest import TestCase
from Simulators.Simulator import Simulator


class TestSimulator(TestCase):

    def setUp(self) -> None:
        num_steps = 10
        random_seed = 0
        debug = True
        self.simulator = Simulator(num_steps=num_steps, random_seed=random_seed, debug=debug)

    def test_run_simulator(self) -> None:
        df = self.simulator.run()
        df
