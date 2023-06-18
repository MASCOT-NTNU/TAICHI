"""
This script runs the replicate study using two different agents in the same field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28

"""
from Simulators.Simulator import Simulator
from usr_func.checkfolder import checkfolder
from joblib import Parallel, delayed
import numpy as np
import os


num_replicates = 2
num_cores = 2
num_steps = 10

seeds = np.random.choice(10000, num_replicates, replace=False)

datapath = os.getcwd() + "/../mafia2_simulation_result/"


def run_replicates(i: int = 0):
    """
    This function runs the replicate study using two different agents in the same field.
    """
    print("seed: ", seeds[i])
    print("replicate: ", i)
    folderpath = datapath + "R_{:03d}/".format(i)
    checkfolder(folderpath)

    simulator = Simulator(num_steps=num_steps, random_seed=seeds[i])
    df = simulator.run()
    df.to_csv(folderpath + "metrics.csv", index=False)


if __name__ == "__main__":
    Parallel(n_jobs=num_cores)(delayed(run_replicates)(i) for i in range(num_replicates))
