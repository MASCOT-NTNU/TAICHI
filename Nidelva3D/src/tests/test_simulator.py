"""
Test the simulator module.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28
"""
from unittest import TestCase
from Simulators.Simulator import Simulator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

        rmse_static = np.stack((df["rmse_grf_static"], df["rmse_gmrf_static"]), axis=1)
        rmse_temporal = np.stack((df["rmse_grf_temporal"], df["rmse_gmrf_temporal"]), axis=1)

        ce_static = np.stack((df["ce_grf_static"], df["ce_gmrf_static"]), axis=1)
        ce_temporal = np.stack((df["ce_grf_temporal"], df["ce_gmrf_temporal"]), axis=1)

        auc_static = np.stack((df["auc_grf_static"], df["auc_gmrf_static"]), axis=1)
        auc_temporal = np.stack((df["auc_grf_temporal"], df["auc_gmrf_temporal"]), axis=1)

        fig = plt.figure(figsize=(40, 16))
        gs = GridSpec(2, 4, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(ibv[:, 0], label="GRF")
        ax.plot(ibv[:, 1], label="GMRF")
        # plt.plot(ibv[:, 0]/np.amax(ibv[:, 0]), label="GRF")
        # plt.plot(ibv[:, 1]/np.amax(ibv[:, 1]), label="GMRF")
        plt.legend()
        plt.title("IBV")

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(vr[:, 0], label="GRF")
        ax.plot(vr[:, 1], label="GMRF")
        # plt.plot(vr[:, 0]/np.amax(vr[:, 0]), label="GRF")
        # plt.plot(vr[:, 1]/np.amax(vr[:, 1]), label="GMRF")
        plt.legend()
        plt.title("VR")

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(rmse_static[:, 0], label="GRF")
        ax.plot(rmse_static[:, 1], label="GMRF")
        # plt.plot(rmse[:, 0]/np.amax(rmse[:, 0]), label="GRF")
        # plt.plot(rmse[:, 1]/np.amax(rmse[:, 1]), label="GMRF")
        plt.legend()
        plt.title("RMSE_static")

        ax = fig.add_subplot(gs[0, 3])
        ax.plot(rmse_temporal[:, 0], label="GRF")
        ax.plot(rmse_temporal[:, 1], label="GMRF")
        # plt.plot(rmse[:, 0]/np.amax(rmse[:, 0]), label="GRF")
        # plt.plot(rmse[:, 1]/np.amax(rmse[:, 1]), label="GMRF")
        plt.legend()
        plt.title("RMSE_temporal")

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(ce_static[:, 0], label="GRF")
        ax.plot(ce_static[:, 1], label="GMRF")
        # plt.plot(ce[:, 0]/np.amax(ce[:, 0]), label="GRF")
        # plt.plot(ce[:, 1]/np.amax(ce[:, 1]), label="GMRF")
        plt.legend()
        plt.title("CE_static")

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(ce_temporal[:, 0], label="GRF")
        ax.plot(ce_temporal[:, 1], label="GMRF")
        # plt.plot(ce[:, 0]/np.amax(ce[:, 0]), label="GRF")
        # plt.plot(ce[:, 1]/np.amax(ce[:, 1]), label="GMRF")
        plt.legend()
        plt.title("CE_temporal")

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(auc_static[:, 0], label="GRF")
        ax.plot(auc_static[:, 1], label="GMRF")
        # plt.plot(auc[:, 0]/np.amax(auc[:, 0]), label="GRF")
        # plt.plot(auc[:, 1]/np.amax(auc[:, 1]), label="GMRF")
        plt.legend()
        plt.title("AUC_static")

        ax = fig.add_subplot(gs[1, 3])
        ax.plot(auc_temporal[:, 0], label="GRF")
        ax.plot(auc_temporal[:, 1], label="GMRF")
        # plt.plot(auc[:, 0]/np.amax(auc[:, 0]), label="GRF")
        # plt.plot(auc[:, 1]/np.amax(auc[:, 1]), label="GMRF")
        plt.legend()
        plt.title("AUC_temporal")

        plt.show()
        print("stop")

