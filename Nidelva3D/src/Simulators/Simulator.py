"""
Simulator generates the simulation result for each agent.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-28

Methodology:
    1. It generates one ground truth field using GRF and SINMOD.
    2. It uses two agent to explore the field.
    3. It observes the data during the simulation and calculates the metrics during the simulation.
    4. It saves the data and metrics to the disk.
"""


from Agents.Agent_adaptive import Agent
import numpy as np
import pandas as pd


class Simulator:

    def __init__(self, num_steps: int = 5, random_seed: int = 0, temporal_truth: bool = True, debug: bool = False) -> None:
        self.grf_agent = Agent(kernel="GRF", num_steps=num_steps, random_seed=random_seed,
                               temporal_truth=temporal_truth, debug=debug)
        self.gmrf_agent = Agent(kernel="GMRF", num_steps=num_steps, random_seed=random_seed,
                                temporal_truth=temporal_truth, debug=debug)

    def run(self) -> 'pd.DataFrame':
        self.grf_agent.run()
        self.gmrf_agent.run()
        self.ibv_grf, self.vr_grf, self.rmse_grf = self.grf_agent.get_metrics()
        self.ibv_gmrf, self.vr_gmrf, self.rmse_gmrf = self.gmrf_agent.get_metrics()
        df = np.hstack((self.ibv_grf, self.vr_grf, self.rmse_grf, self.ibv_gmrf, self.vr_gmrf, self.rmse_gmrf))
        df = pd.DataFrame(df, columns=["ibv_grf", "vr_grf", "rmse_grf", "ibv_gmrf", "vr_gmrf", "rmse_gmrf"])
        return df


if __name__ == "__main__":
    s = Simulator()
