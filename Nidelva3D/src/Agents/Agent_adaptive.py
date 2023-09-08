"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

The goal of the agent is to conduct the autonomous sampling operation by using the following procedure:
- Sense
- Plan
- Act

Sense refers to the in-situ measurements. Once the agent obtains the sampled values in the field. Then it can plan based
on the updated knowledge for the field. Therefore, it can act according to the planned manoeuvres.
"""
from Planner.Myopic3D import Myopic3D
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.Visualiser_myopic import Visualiser
from usr_func.checkfolder import checkfolder
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import norm
import numpy as np
import time
import os


class Agent:

    __loc_start = np.array([2500, 1000, 0.5])   # s1, starting location
    # __loc_start = np.array([2200, 800, 0.5])   # s1, starting location
    # __loc_start = np.array([3500, 800, 0.5])   # s1, starting location
    __counter = 0

    def __init__(self, kernel: str = "GMRF", num_steps: int = 5,
                 random_seed: int = 0, temporal_truth: bool = True, debug: bool = True) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        self.temporal_truth = temporal_truth

        print("Number of steps: ", num_steps)
        self.__num_steps = num_steps

        # s1: setup planner.
        self.myopic = Myopic3D(kernel=kernel)
        self.grid = self.myopic.kernel.get_grid()

        # s2: setup AUV simulator.
        self.auv = AUVSimulator(random_seed=random_seed, temporal_truth=temporal_truth)
        ctd = self.auv.ctd
        if self.temporal_truth:
            self.mu_truth = ctd.get_salinity_at_dt_loc(dt=0, loc=self.grid)
        else:
            self.mu_truth = ctd.get_salinity_at_loc(loc=self.grid)
        self.mu_truth[self.mu_truth < 0] = 0

        # s3, set up the metrics.
        self.__threshold = self.myopic.kernel.get_threshold()
        self.__ibv = np.zeros([self.__num_steps, 1])
        self.__vr = np.zeros([self.__num_steps, 1])
        self.__rmse_temporal = np.zeros([self.__num_steps, 1])
        self.__rmse_static = np.zeros([self.__num_steps, 1])
        self.__ce_temporal = np.zeros([self.__num_steps, 1])
        self.__ce_static = np.zeros([self.__num_steps, 1])
        self.__corr_coef = np.zeros([self.__num_steps, 1])
        self.__auc_temporal = np.zeros([self.__num_steps, 1])
        self.__auc_static = np.zeros([self.__num_steps, 1])
        self.update_metrics()

        self.debug = debug
        if self.debug:
            # s3: setup Visualiser.
            figpath = os.getcwd() + "/../fig/Myopic3D/" + str(random_seed) + "/" + kernel + "/"
            checkfolder(figpath)

            truth = self.mu_truth
            grid = self.grid
            ind_surface = np.where(grid[:, 2] == 0.5)[0]
            truth_surface = truth[ind_surface]

            import matplotlib.pyplot as plt
            from matplotlib.pyplot import get_cmap
            plt.figure()
            plt.scatter(grid[ind_surface, 1], grid[ind_surface, 0], c=truth_surface, cmap=get_cmap("BrBG", 10),
                        vmin=0, vmax=33)
            plt.colorbar()
            plt.savefig(figpath + 'truth.png')
            plt.close("all")

            self.visualiser = Visualiser(self, figpath=figpath)

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # c1: start the operation from scratch.
        # id_start = np.random.randint(0, len(self.myopic.waypoint_graph.get_waypoints()))
        id_start = self.myopic.waypoint_graph.get_ind_from_waypoint(self.__loc_start)
        id_curr = id_start

        # s1: setup the planner -> only once
        self.myopic.set_current_index(id_curr)
        self.myopic.set_next_index(id_curr)
        self.myopic.set_pioneer_index(id_curr)

        # a1: move to current location
        self.auv.move_to_location(self.myopic.waypoint_graph.get_waypoint_from_ind(id_curr))

        t_start = time.time()
        t_pop_last = time.time()

        if self.debug:
            self.visualiser.plot_agent()

        while True:
            t_end = time.time()
            """
            Simulating the AUV behaviour, not the actual one
            """
            t_gap = t_end - t_start
            if t_gap >= 5:
                self.auv.arrive()
                t_start = time.time()

            if self.__counter == 0:
                if t_end - t_pop_last >= 50:
                    self.auv.popup()
                    print("POP UP")
                    t_pop_last = time.time()

            if self.auv.is_arrived():
                if t_end - t_pop_last >= 20:
                    self.auv.popup()
                    print("POPUP")
                    t_pop_last = time.time()

                if self.__counter == 0:
                    # s2: get next index using get_pioneer_waypoint
                    ind = self.myopic.get_pioneer_waypoint_index()
                    self.myopic.set_next_index(ind)

                    # p1: parallel move AUV to the first location
                    loc = self.myopic.waypoint_graph.get_waypoint_from_ind(ind)
                    self.auv.move_to_location(loc)

                    # s3: update planner -> so curr and next waypoint is updated
                    self.myopic.update_planner()

                    # s4: get pioneer waypoint
                    self.myopic.get_pioneer_waypoint_index()

                    # # s5: obtain CTD data
                    ctd_data = self.auv.get_ctd_data()

                    # # s5: assimilate data
                    self.myopic.kernel.assimilate_data(ctd_data)
                else:
                    ind = self.myopic.get_current_index()
                    loc = self.myopic.waypoint_graph.get_waypoint_from_ind(ind)
                    self.auv.move_to_location(loc)

                    # a1: gather AUV data
                    ctd_data = self.auv.get_ctd_data()

                    # a2: update GMRF field
                    self.myopic.kernel.assimilate_data(ctd_data)

                    # ss2: update planner
                    self.myopic.update_planner()

                    # ss3: plan ahead.
                    self.myopic.get_pioneer_waypoint_index()

                    if self.__counter == self.__num_steps:
                        print("Mission complete!")
                        break

                print("counter: ", self.__counter)
                print(self.myopic.get_current_index())
                print(self.myopic.get_trajectory_indices())
                self.update_metrics()
                self.__counter += 1

                if self.debug:
                    self.visualiser.plot_agent()

    def update_metrics(self) -> None:
        mu = self.myopic.kernel.get_mu()
        sigma_diag = self.myopic.kernel.get_mvar()
        self.__ibv[self.__counter] = self.__get_ibv(self.__threshold, mu, sigma_diag)
        self.__vr[self.__counter] = np.sum(sigma_diag)

        mu[mu < 0] = 0

        # s0, update static metrics
        self.__rmse_static[self.__counter] = mean_squared_error(self.mu_truth.flatten(), mu.flatten(), squared=False)
        self.__ce_static[self.__counter] = self.__cal_classification_error(self.mu_truth.flatten(),
                                                                             mu.flatten(), self.__threshold)
        self.__auc_static[self.__counter] = self.__cal_auc_roc(self.__threshold, mu, self.mu_truth, sigma_diag)

        # s1, update temporal metrics
        self.mu_truth = self.auv.ctd.get_salinity_at_dt_loc(dt=0, loc=self.grid)
        self.__rmse_temporal[self.__counter] = mean_squared_error(self.mu_truth.flatten(), mu.flatten(), squared=False)
        self.__corr_coef[self.__counter] = self.__cal_corr_coef(self.mu_truth.flatten(), mu.flatten())
        self.__ce_temporal[self.__counter] = self.__cal_classification_error(self.mu_truth.flatten(),
                                                                             mu.flatten(), self.__threshold)
        self.__auc_temporal[self.__counter] = self.__cal_auc_roc(self.__threshold, mu, self.mu_truth, sigma_diag)

    @staticmethod
    def __cal_corr_coef(x, y) -> float:
        """
        Calculate the correlation coefficient between two vectors.

        Method:
        1. Calculate the covariance matrix between two vectors.
        2. Calculate the correlation coefficient.

        """
        d1 = x - np.mean(x)
        d2 = y - np.mean(y)
        cov = d1.T @ d2
        corr = cov / np.linalg.norm(d1) / np.linalg.norm(d2)
        return corr

    @staticmethod
    def __cal_classification_error(x, y, threshold) -> float:
        """
        Calculate the classification error between two vectors.
        """
        X = np.where(x > threshold, 1, 0)
        Y = np.where(y > threshold, 1, 0)
        CE = np.sum(np.abs(X - Y)) / len(X)
        return CE

    @staticmethod
    def __cal_auc_roc(threshold: float, mu: np.ndarray, mu_truth: np.ndarray, sigma_diag: np.ndarray) -> float:
        """
        Calculate the area under the curve between two vectors.
        """
        p = norm.cdf(threshold, mu.squeeze(), np.sqrt(sigma_diag))
        truth_labels = np.where(mu_truth < threshold, 1, 0)
        auc_roc = roc_auc_score(truth_labels, p)
        return auc_roc

    @staticmethod
    def __get_ibv(threshold: float, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
        """ !!! Be careful with dimensions, it can lead to serious problems.
        !!! Be careful with standard deviation is not variance, so it does not cause significant issues tho.
        :param mu: n x 1 dimension
        :param sigma_diag: n x 1 dimension
        :return:
        """
        p = norm.cdf(threshold, mu.squeeze(), np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def get_counter(self):
        return self.__counter

    def get_metrics(self) -> tuple:
        return self.__ibv, self.__vr, self.__rmse_temporal, self.__corr_coef, self.__ce_temporal, \
            self.__auc_temporal, self.__rmse_static, self.__ce_static, self.__auc_static


if __name__ == "__main__":
    a = Agent()
    a.run()


