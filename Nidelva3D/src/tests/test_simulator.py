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
        num_steps = 30
        random_seed = 25
        debug = True
        temporal_truth = True
        self.simulator = Simulator(num_steps=num_steps, random_seed=random_seed,
                                   temporal_truth=temporal_truth, debug=debug)

    def test_run_simulator(self) -> None:
        df = self.simulator.run()
        ibv = np.stack((df["ibv_grf"], df["ibv_gmrf"]), axis=1)
        vr = np.stack((df["vr_grf"], df["vr_gmrf"]), axis=1)
        rmse = np.stack((df["rmse_grf"], df["rmse_gmrf"]), axis=1)
        corr = np.stack((df["corr_grf"], df["corr_gmrf"]), axis=1)
        ce = np.stack((df["ce_grf"], df["ce_gmrf"]), axis=1)
        auc = np.stack((df["auc_grf"], df["auc_gmrf"]), axis=1)

        plt.figure(figsize=(40, 8))
        plt.subplot(151)
        plt.plot(ibv[:, 0], label="GRF")
        plt.plot(ibv[:, 1], label="GMRF")
        # plt.plot(ibv[:, 0]/np.amax(ibv[:, 0]), label="GRF")
        # plt.plot(ibv[:, 1]/np.amax(ibv[:, 1]), label="GMRF")
        plt.legend()
        plt.title("IBV")
        plt.subplot(152)
        plt.plot(vr[:, 0], label="GRF")
        plt.plot(vr[:, 1], label="GMRF")
        # plt.plot(vr[:, 0]/np.amax(vr[:, 0]), label="GRF")
        # plt.plot(vr[:, 1]/np.amax(vr[:, 1]), label="GMRF")
        plt.legend()
        plt.title("VR")
        plt.subplot(153)
        plt.plot(rmse[:, 0], label="GRF")
        plt.plot(rmse[:, 1], label="GMRF")
        # plt.plot(rmse[:, 0]/np.amax(rmse[:, 0]), label="GRF")
        # plt.plot(rmse[:, 1]/np.amax(rmse[:, 1]), label="GMRF")
        plt.legend()
        plt.title("RMSE")
        plt.subplot(154)
        plt.plot(auc[:, 0], label="GRF")
        plt.plot(auc[:, 1], label="GMRF")
        plt.legend()
        plt.title("AUC")
        plt.subplot(155)
        plt.plot(ce[:, 0], label="GRF")
        plt.plot(ce[:, 1], label="GMRF")
        plt.legend()
        plt.title("CE")
        plt.show()
        print("stop")

        df
