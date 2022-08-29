"""
This config file contains simulation replicate study results
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-16
"""

from TAICHI.Square2D.Config.Config import NUM_STEPS
from usr_func import np, vectorise


class SimulationResultContainer:

    def __init__(self, strategyname):
        self.strategyname = strategyname
        self.ibv = np.empty([0, NUM_STEPS])
        self.rmse = np.empty([0, NUM_STEPS])
        self.uncertainty = np.empty([0, NUM_STEPS])
        self.crps = np.empty([0, NUM_STEPS])

    def append(self, ag):
        rmse = ag.rmse
        uncertainty = ag.uncertainty
        ibv = ag.ibv
        crps = ag.crps
        self.rmse = np.append(self.rmse, vectorise(rmse).T, axis=0)
        self.uncertainty = np.append(self.uncertainty, vectorise(uncertainty).T, axis=0)
        self.ibv = np.append(self.ibv, vectorise(ibv).T, axis=0)
        self.crps = np.append(self.crps, vectorise(crps).T, axis=0)


