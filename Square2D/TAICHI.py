"""
This script simulates TAICHI's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""
import matplotlib.pyplot as plt
import pandas as pd

from usr_func import *
from TAICHI.Square2D.Config.Config import *
from TAICHI.Square2D.Agent import Agent
from TAICHI.Square2D.SimulationResultContainer import SimulationResultContainer as SRC


class TAICHI:

    def __init__(self):
        self.middle_point = [0, 0]
        self.radius_skp = LOITER_RADIUS + SAFETY_DISTANCE  # station-keeping radius
        print("Hello, this is TAICHI")

    def setup_agents(self, ag1_loc, ag2_loc, seed=None):
        self.ag1_name = "A1"
        self.ag2_name = "A2"
        self.ag1 = Agent(self.ag1_name, seed=seed)
        self.ag1.set_starting_location(ag1_loc)
        self.ag1.prepare_run()
        self.ag2 = Agent(self.ag2_name, seed=seed)
        self.ag2.set_starting_location(ag2_loc)
        self.ag2.prepare_run()

    def update_compass(self, ag1_loc=None, ag2_loc=None):  # remember that it is different in xy-coord and wgs-coord
        if ag1_loc is None:
            ag1_loc = [0, 0]
        if ag2_loc is None:
            ag2_loc = [0, 0]
        self.middle_point = [(ag1_loc[0] + ag2_loc[0]) / 2, (ag1_loc[1] + ag2_loc[1]) / 2]
        self.ag1_angle_to_middle_point = np.math.atan2(ag1_loc[1] - self.middle_point[1],
                                                       ag1_loc[0] - self.middle_point[0])
        self.ag2_angle_to_middle_point = np.math.atan2(ag2_loc[1] - self.middle_point[1],
                                                       ag2_loc[0] - self.middle_point[0])

        self.ag1_skp_loc = [self.middle_point[0] + self.radius_skp * np.cos(self.ag1_angle_to_middle_point),
                            self.middle_point[1] + self.radius_skp * np.sin(self.ag1_angle_to_middle_point)]
        self.ag2_skp_loc = [self.middle_point[0] + self.radius_skp * np.cos(self.ag2_angle_to_middle_point),
                            self.middle_point[1] + self.radius_skp * np.sin(self.ag2_angle_to_middle_point)]

    def repartition(self):
        self.waypoint_graph = self.ag1.waypoints
        self.update_compass(self.ag1.waypoints[self.ag1.ind_current_waypoint],
                            self.ag2.waypoints[self.ag2.ind_current_waypoint])


        self.dx = self.waypoint_graph[:, 0] - self.middle_point[0]
        self.dy = self.waypoint_graph[:, 1] - self.middle_point[1]

        vec_middle_to_ag1 = vectorise([self.ag1.waypoints[self.ag1.ind_current_waypoint, 0] - self.middle_point[0],
                                       self.ag1.waypoints[self.ag1.ind_current_waypoint, 1] - self.middle_point[1]])
        self.vec_prod = np.vstack((self.dx, self.dy)).T @ vec_middle_to_ag1

        self.ind_legal = np.arange(self.ag1.waypoints.shape[0])
        self.same_side = (self.vec_prod > 0).flatten()
        self.ind_ag1 = self.ind_legal[self.same_side]
        self.ind_ag2 = self.ind_legal[~self.same_side]

        return self.ind_ag1, self.ind_ag2
        # plt.plot(self.waypoint_graph[:, 0], self.waypoint_graph[:, 1], 'k.', alpha=.3)
        # plt.plot(self.ag1.waypoints[self.ag1.ind_current_waypoint, 0],
        #          self.ag1.waypoints[self.ag1.ind_current_waypoint, 1], 'r.', alpha=.3, label="AG1 Old")
        # plt.plot(self.ag2.waypoints[self.ag2.ind_current_waypoint, 0],
        #          self.ag2.waypoints[self.ag2.ind_current_waypoint, 1], 'b.', alpha=.3, label="AG2 Old")
        # plt.plot(self.ag1_skp_loc[0], self.ag1_skp_loc[1], 'r.', label='AG1')
        # plt.plot(self.ag2_skp_loc[0], self.ag2_skp_loc[1], 'b.', label='AG2')
        # plt.plot(self.waypoint_graph[self.ind_ag1, 0], self.waypoint_graph[self.ind_ag1, 1], 'r*', alpha=.1)
        # plt.plot(self.waypoint_graph[self.ind_ag2, 0], self.waypoint_graph[self.ind_ag2, 1], 'b*', alpha=.1)
        # # plt.legend()
        # filename = FIGPATH + "Partition/P_{:03d}.jpg".format(i)
        # checkfolder(os.path.dirname(filename))
        # plt.savefig(filename)
        # # plt.show()
        # plt.close("all")

    def run_twin_agents(self, ag1, ag2):
        for i in range(NUM_STEPS):
            print("Step: ", i)
            share = False
            pre_share = False

            t1 = time.time()
            ag1.sample()                                     # step 1
            ag2.sample()

            if (i + 1) % DATA_SHARING_GAP == 0:
                pre_share = True
            elif i > 0 and i % DATA_SHARING_GAP == 0:
                share = True

                self.update_compass(self.ag1.waypoints[self.ag1.ind_current_waypoint, :],
                                    self.ag2.waypoints[self.ag2.ind_current_waypoint, :])
                self.repartition()  # step 0, repartition

                # save data from agent1, agent2
                ag1.save_agent_data()                        # step 2
                ag2.save_agent_data()
                # load data from agent1, agent2
                ag1.load_data_from_agents(ag2)
                ag2.load_data_from_agents(ag1)

            ag1.run(step=i, pre_share=pre_share, share=share, other_agent=ag2, ind_legal=self.ind_ag1)  # step 4
            ag2.run(step=i, pre_share=pre_share, share=share, other_agent=ag1, ind_legal=self.ind_ag2)  # step 4

            if share:
                ag1.clear_agent_data()
                ag2.clear_agent_data()

            t2 = time.time()
            print("One step running takes: ", t2 - t1)

    def run_simulator(self, replicates=1):
        self.result_taichi = SRC("TAICHI")
        self.result_monk = SRC("Monk")

        for i in range(replicates):
            t_start = time.time()
            print("replicate: ", i)
            seed = np.random.randint(10000)
            print("seed: ", seed)

            blockPrint()
            self.ag1 = Agent("TAICHI_YIN", seed=seed)
            self.ag1.set_starting_location([0, 1])
            self.ag1.prepare_run()
            self.ag2 = Agent("TAICHI_YANG", seed=seed)
            self.ag2.set_starting_location([1, 0])
            self.ag2.prepare_run()

            self.ag3 = Agent("MONK", seed=seed)
            self.ag3.set_starting_location([0, 1])
            self.ag3.prepare_run()

            for j in range(NUM_STEPS):
                # enablePrint()
                print("Step: ", j)
                share = False
                pre_share = False
                # blockPrint()

                t1 = time.time()
                self.ag1.sample()  # step 1
                self.ag2.sample()
                self.ag3.sample()

                self.ind_ag1 = None
                self.ind_ag2 = None

                if (j + 1) % DATA_SHARING_GAP == 0:
                    pre_share = True
                elif j > 0 and j % DATA_SHARING_GAP == 0:
                    share = True

                    self.update_compass(self.ag1.waypoints[self.ag1.ind_current_waypoint, :],
                                        self.ag2.waypoints[self.ag2.ind_current_waypoint, :])
                    self.ind_ag1, self.ind_ag2 = self.repartition()  # step 0, repartition

                    # save data from agent1, agent2
                    self.ag1.save_agent_data()  # step 2
                    self.ag2.save_agent_data()
                    # load data from agent1, agent2
                    self.ag1.load_data_from_agents(self.ag2)
                    self.ag2.load_data_from_agents(self.ag1)

                self.ag1.run(step=i, pre_share=pre_share, share=share, other_agent=self.ag2, ind_legal=self.ind_ag1)  # step 4
                self.ag2.run(step=i, pre_share=pre_share, share=share, other_agent=self.ag1, ind_legal=self.ind_ag2)  # step 4
                self.ag3.run(step=i)

                if share:
                    self.ag1.clear_agent_data()
                    self.ag2.clear_agent_data()

                t2 = time.time()
                print("One step running takes: ", t2 - t1)
            self.result_taichi.append(self.ag1)
            self.result_monk.append(self.ag3)
            enablePrint()
            t_end = time.time()
            print("Time consumed: ", t_end - t_start)
        self.save_simulation_result()

    def save_simulation_result(self):
        df = pd.DataFrame(self.result_taichi.rmse)
        df.to_csv(FILEPATH + "SimulationResult/TAICHI_RMSE.csv", index=False)

        df = pd.DataFrame(self.result_taichi.ibv)
        df.to_csv(FILEPATH + "SimulationResult/TAICHI_IBV.csv", index=False)

        df = pd.DataFrame(self.result_taichi.uncertainty)
        df.to_csv(FILEPATH + "SimulationResult/TAICHI_UNCERTAINTY.csv", index=False)

        df = pd.DataFrame(self.result_taichi.crps)
        df.to_csv(FILEPATH + "SimulationResult/TAICHI_CRPS.csv", index=False)

        df = pd.DataFrame(self.result_monk.rmse)
        df.to_csv(FILEPATH + "SimulationResult/MONK_RMSE.csv", index=False)

        df = pd.DataFrame(self.result_monk.ibv)
        df.to_csv(FILEPATH + "SimulationResult/MONK_IBV.csv", index=False)

        df = pd.DataFrame(self.result_monk.uncertainty)
        df.to_csv(FILEPATH + "SimulationResult/MONK_UNCERTAINTY.csv", index=False)

        df = pd.DataFrame(self.result_monk.crps)
        df.to_csv(FILEPATH + "SimulationResult/MONK_CRPS.csv", index=False)

    def check_taichi(self):
        a1 = [0, 1]
        a2 = [1, 0]
        self.setup_agents(a1, a2, seed=0)
        self.update_compass(a1, a2)
        self.repartition()
        self.run_twin_agents(self.ag1, self.ag2)

    def run_1_agent(self):
        self.ag1 = Agent(self.ag1_name, seed=seed)
        self.ag1.set_starting_location(ag1_loc)
        self.ag1.prepare_run()
        for i in range(NUM_STEPS):
            ag.sample()
            ag.run(step=i)

    def check_src(self):

        # x = np.arange(tc.result_taichi.crps.shape[1]-1)
        # y = np.mean(tc.result_taichi.crps[:, 1:], axis=0)
        # yerr = np.std(tc.result_taichi.crps[:, 1:], axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="TAICHI")
        #
        # x = np.arange(tc.result_monk.crps.shape[1]-1)
        # y = np.mean(tc.result_monk.crps[:, 1:], axis=0)
        # yerr = np.std(tc.result_monk.crps[:, 1:], axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="MONK")
        # plt.legend()
        # plt.title("CRPS")
        # plt.show()

        # plt.plot(tc.result_taichi.crps[0, 1:], label="TAICHI")
        # plt.plot(tc.result_monk.crps[0, 1:], label="MONK")
        # plt.legend()
        # plt.title("CRPS")
        # plt.show()

        # x = np.arange(tc.result_taichi.ibv.shape[1]-1)
        # y = np.mean(tc.result_taichi.ibv[:, 1:], axis=0)
        # yerr = np.std(tc.result_taichi.ibv[:, 1:], axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="TAICHI")
        #
        # x = np.arange(tc.result_monk.ibv.shape[1]-1)
        # y = np.mean(tc.result_monk.ibv[:, 1:], axis=0)
        # yerr = np.std(tc.result_monk.ibv[:, 1:], axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="MONK")
        # plt.legend()
        # plt.title("IBV")
        # plt.show()

        x = np.arange(tc.result_taichi.uncertainty.shape[1] - 1)
        Y = np.log(tc.result_taichi.uncertainty[:, 1:])
        y = np.mean(Y, axis=0)
        yerr = np.std(Y, axis=0)
        plt.errorbar(x, y, yerr=yerr, label="TAICHI")

        x = np.arange(tc.result_monk.uncertainty.shape[1] - 1)
        Y = np.log(tc.result_monk.uncertainty[:, 1:])
        y = np.mean(Y, axis=0)
        yerr = np.std(Y, axis=0)
        plt.errorbar(x, y, yerr=yerr, label="MONK")
        plt.legend()
        plt.title("Uncertainty")
        plt.show()

        # x = np.arange(tc.result_taichi.rmse.shape[1])
        # y = np.mean(tc.result_taichi.rmse, axis=0)
        # yerr = np.std(tc.result_taichi.rmse, axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="TAICHI")
        #
        # x = np.arange(tc.result_monk.rmse.shape[1])
        # y = np.mean(tc.result_monk.rmse, axis=0)
        # yerr = np.std(tc.result_monk.rmse, axis=0)
        # plt.errorbar(x, y, yerr=yerr, label="MONK")
        #
        # # plt.plot(tc.result_monk.rmse[:, 1:], label="MONK")
        # plt.legend()
        # plt.title("RMSE")
        # plt.show()

        import numpy as np
        from scipy.spatial import Voronoi, voronoi_plot_2d
        import shapely.geometry
        import shapely.ops

        points = np.random.random((10, 2))
        vor = Voronoi(points)
        voronoi_plot_2d(vor)
        plt.show()

        lines = [
            shapely.geometry.LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line
        ]

        plt.plot(lines[0])
        plt.show()


if __name__ == "__main__":
    tc = TAICHI()
    # tc.check_taichi()
    # tc.run()
    tc.run_simulator(50)






