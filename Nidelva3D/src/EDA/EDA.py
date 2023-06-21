"""
EDA visualizes the conditional field over sampling.
"""
from GMRF.GMRF import GMRF
from Planner.Myopic3D import Myopic3D
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.Visualiser_myopic import Visualiser
import numpy as np
import pandas as pd
import time
import os


class Agent:
    # __loc_start = np.array([6876.20208333, 4549.81267591, -.5])
    # __loc_end = np.array([0, 0, 0])
    # __NUM_STEP = 50
    __counter = 0
    trajectory = np.empty([0, 3])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner
        self.myopic = Myopic3D(kernel="GMRF")

        # s2: load data
        self.datapath = "/Users/yaolin/Library/CloudStorage/OneDrive-NTNU/MASCOT_PhD/Data/Nidelva/20230621/TAICHI/GMRF/raw_ctd/1687334189/"
        self.files = os.listdir(self.datapath)
        self.files.sort()
        print(self.files)

        # df = pd.read_csv("csv/")

        # s3: setup Visualiser.
        self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/TAICHI/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # c1: start the operation from scratch.
        self.visualiser.plot_agent()

        for i in range(len(self.files)):
            # a1: gather AUV data
            ctd_data = pd.read_csv(self.datapath + self.files[i]).to_numpy()[:, 1:]
            self.trajectory = np.append(self.trajectory, ctd_data[:, :-1], axis=0)

            # a2: update GMRF field
            self.myopic.kernel.assimilate_data(ctd_data)
            print("counter: ", self.__counter)
            self.visualiser.plot_agent()
            self.__counter += 1

    def get_counter(self):
        return self.__counter

    def get_trajectory(self) -> np.ndarray:
        return self.trajectory


if __name__ == "__main__":
    a = Agent()
    a.run()


