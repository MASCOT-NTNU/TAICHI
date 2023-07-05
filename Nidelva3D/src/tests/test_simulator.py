"""
Test the simulator module.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28
"""
from unittest import TestCase
from Simulators.Simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np


class TestSimulator(TestCase):

    def setUp(self) -> None:
        num_steps = 15
        random_seed = 14
        debug = True
        self.simulator = Simulator(num_steps=num_steps, random_seed=random_seed, debug=debug)

    def test_run_simulator(self) -> None:
        df = self.simulator.run()
        ibv = np.stack((df["ibv_grf"], df["ibv_gmrf"]), axis=1)
        vr = np.stack((df["vr_grf"], df["vr_gmrf"]), axis=1)
        rmse = np.stack((df["rmse_grf"], df["rmse_gmrf"]), axis=1)

        plt.figure(figsize=(25, 8))
        plt.subplot(131)
        plt.plot(ibv[:, 0]/np.amax(ibv[:, 0]), label="GRF")
        plt.plot(ibv[:, 1]/np.amax(ibv[:, 1]), label="GMRF")
        plt.legend()
        plt.title("IBV")
        plt.subplot(132)
        plt.plot(vr[:, 0]/np.amax(vr[:, 0]), label="GRF")
        plt.plot(vr[:, 1]/np.amax(vr[:, 1]), label="GMRF")
        plt.legend()
        plt.title("VR")
        plt.subplot(133)
        plt.plot(rmse[:, 0], label="GRF")
        plt.plot(rmse[:, 1], label="GMRF")
        # plt.plot(rmse[:, 0]/np.amax(rmse[:, 0]), label="GRF")
        # plt.plot(rmse[:, 1]/np.amax(rmse[:, 1]), label="GMRF")
        plt.legend()
        plt.title("RMSE")
        plt.show()
        print("stop")

        df
