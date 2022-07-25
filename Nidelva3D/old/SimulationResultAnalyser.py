"""
This script analyses the simulation result
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-19
"""

from usr_func import *
from TAICHI.Nidelva3D.Config.Config import *


class SRA:

    def __init__(self):
        pass

    def load_simulation_result(self):
        self.taichi_rmse = pd.read_csv(FILEPATH + "SimulationResult/TAICHI_RMSE.csv").to_numpy()
        self.taichi_ibv = pd.read_csv(FILEPATH + "SimulationResult/TAICHI_IBV.csv").to_numpy()
        self.taichi_uncertainty = pd.read_csv(FILEPATH + "SimulationResult/TAICHI_UNCERTAINTY.csv").to_numpy()
        self.taichi_crps = pd.read_csv(FILEPATH + "SimulationResult/TAICHI_CRPS.csv").to_numpy()

        self.monk_rmse = pd.read_csv(FILEPATH + "SimulationResult/MONK_RMSE.csv").to_numpy()
        self.monk_ibv = pd.read_csv(FILEPATH + "SimulationResult/MONK_IBV.csv").to_numpy()
        self.monk_uncertainty = pd.read_csv(FILEPATH + "SimulationResult/MONK_UNCERTAINTY.csv").to_numpy()
        self.monk_crps = pd.read_csv(FILEPATH + "SimulationResult/MONK_CRPS.csv").to_numpy()
        print("Simulation result is loaded successfully!")

    def plot_simulation_result(self):
        x = np.arange(self.taichi_ibv.shape[1]-1)
        fig = plt.figure(figsize=(30, 8))
        gs = GridSpec(nrows=1, ncols=3)
        ax1 = fig.add_subplot(gs[0])
        ax1.errorbar(x, np.mean(self.monk_ibv[:, 1:], axis=0), yerr=np.std(self.monk_ibv[:, 1:], axis=0), fmt="-o", capsize=5,
                     label="1-Agent")
        ax1.errorbar(x, np.mean(self.taichi_ibv[:, 1:], axis=0), yerr=np.std(self.taichi_ibv[:, 1:], axis=0), fmt="-o", capsize=5,
                     label="2-Agents")
        plt.xlabel('Time steps')
        plt.ylabel('IBV')
        plt.legend()

        ax2 = fig.add_subplot(gs[1])
        ax2.errorbar(x, np.mean(self.monk_rmse[:, 1:], axis=0), yerr=np.std(self.monk_rmse[:, 1:], axis=0), fmt="-o", capsize=5,
                     label="1-Agent")
        ax2.errorbar(x, np.mean(self.taichi_rmse[:, 1:], axis=0), yerr=np.std(self.taichi_rmse[:, 1:], axis=0), fmt="-o", capsize=5,
                     label="2-Agents")
        plt.xlabel('Time steps')
        plt.ylabel('RMSE')
        plt.legend()

        ax3 = fig.add_subplot(gs[2])
        ax3.errorbar(x, np.mean(self.monk_uncertainty[:, 1:], axis=0), yerr=np.std(self.monk_uncertainty[:, 1:], axis=0),
                     fmt="-o", capsize=5, label="1-Agent")
        ax3.errorbar(x, np.mean(self.taichi_uncertainty[:, 1:], axis=0), yerr=np.std(self.taichi_uncertainty[:, 1:], axis=0),
                     fmt="-o", capsize=5, label="2-Agents")
        plt.xlabel('Time steps')
        plt.ylabel('Uncertainty')
        plt.legend()
        plt.suptitle("Simulation result for 50 replicates in Nidelva 3D case.")
        plt.savefig(FIGPATH + "SR3.pdf")
        plt.show()
        plt.close('all')

        plt.figure(figsize=(10, 8))
        plt.errorbar(x, np.mean(self.monk_crps[:, 1:], axis=0), yerr=np.std(self.monk_crps[:, 1:], axis=0),
                     fmt="-o", capsize=5, label="1-Agent")
        plt.errorbar(x, np.mean(self.taichi_crps[:, 1:], axis=0), yerr=np.std(self.taichi_crps[:, 1:], axis=0),
                     fmt="-o", capsize=5, label="2-Agents")
        plt.xlabel('Time steps')
        plt.ylabel('CRPS')
        plt.legend()
        plt.savefig(FIGPATH + "SR_CRPS.pdf")
        plt.show()
        plt.close('all')


if __name__ == "__main__":
    s = SRA()
    s.load_simulation_result()
    s.plot_simulation_result()
