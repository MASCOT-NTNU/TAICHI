"""
This script simulates TAICHI's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""

from usr_func import *
from TAICHI.Square2D.Config.Config import *
from TAICHI.Square2D.Agent import Agent
# from TAICHI.Square2D.SimulationResultContainer import SimulationResultContainer as SRC


class TAICHI:

    def __init__(self):
        self.middle_point = [0, 0]
        self.radius_skp = LOITER_RADIUS + SAFETY_DISTANCE  # station-keeping radius
        # self.setup_agents()
        print("Hello, this is TAICHI")

    def setup_agents(self):
        self.ag1_name = "A1"
        self.ag2_name = "A2"
        self.ag1 = Agent(self.ag1_name)
        # self.ag1.set_starting_location(AGENT1_START_LOCATION)
        self.ag1.prepare_run(np.random.randint(self.ag1.waypoints.shape[0]))
        self.ag2 = Agent(self.ag2_name)
        # self.ag2.set_starting_location(AGENT2_START_LOCATION)
        self.ag2.prepare_run(np.random.randint(self.ag1.waypoints.shape[0]))

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

    def split_regions(self):
        self.setup_agents()
        self.update_compass(self.ag1.waypoints[self.ag1.ind_current_waypoint],
                            self.ag2.waypoints[self.ag2.ind_current_waypoint])
        self.waypoint_graph = self.ag1.waypoints

        self.dx = self.waypoint_graph[:, 0] - self.middle_point[0]
        self.dy = self.waypoint_graph[:, 1] - self.middle_point[1]

        vec_middle_to_ag1 = vectorise([self.ag1.waypoints[self.ag1.ind_current_waypoint, 0] - self.middle_point[0],
                                       self.ag1.waypoints[self.ag1.ind_current_waypoint, 1] - self.middle_point[1]])
        self.vecprod = np.vstack((self.dx, self.dy)).T @ vec_middle_to_ag1

        self.ind_ag1 = np.where(self.vecprod > 0)[0]

        plt.plot(self.waypoint_graph[:, 0], self.waypoint_graph[:, 1], 'k.', alpha=.3)
        plt.plot(self.ag1.waypoints[self.ag1.ind_current_waypoint, 0],
                 self.ag1.waypoints[self.ag1.ind_current_waypoint, 1], 'r.', alpha=.3, label="AG1 Old")
        plt.plot(self.ag2.waypoints[self.ag2.ind_current_waypoint, 0],
                 self.ag2.waypoints[self.ag2.ind_current_waypoint, 1], 'b.', alpha=.3, label="AG2 Old")
        plt.plot(self.ag1_skp_loc[0], self.ag1_skp_loc[1], 'r.', label='AG1')
        plt.plot(self.ag2_skp_loc[0], self.ag2_skp_loc[1], 'b.', label='AG2')
        plt.plot(self.waypoint_graph[self.ind_ag1, 0], self.waypoint_graph[self.ind_ag1, 1], 'r*', alpha=.1)
        # plt.legend()
        plt.show()
        pass

    def run_multiple_agents(self, ags):
        for i in range(NUM_STEPS):
            print("Step: ", i)
            share = False
            pre_share = False

            t1 = time.time()

            for ag in ags:
                ag.sample()
            # self.ag1.sample()
            # self.ag2.sample()

            if (i + 2) % DATA_SHARING_GAP == 0:
                # print("only pre share")
                pre_share = True

                # ag1_loc = self.ag1.waypoints[self.ag1.ind_next_waypoint]  # use next waypoint since system requires
                # ag2_loc = self.ag2.waypoints[self.ag2.ind_next_waypoint]  # pre-advanced calculation
                # self.update_universe(ag1_loc, ag2_loc)
                # self.get_taichi()

                # update agent ind_pioneer waypoint to taichi position
                # self.ag1.update_pioneer_waypoint(waypoint_location=self.agent1_new_location)
                # self.ag2.update_pioneer_waypoint(waypoint_location=self.agent2_new_location)

            elif i > 0 and i % DATA_SHARING_GAP == 0:
                # print("now share")
                share = True

                # save data from agent1, agent2
                for ag in ags:
                    ag.save_agent_data()
                # self.ag1.save_agent_data()
                # self.ag2.save_agent_data()

                # load data from agent1, agent2
                for j in range(len(ags)):
                    ags_rest = ags[:j] + ags[j+1:]
                    ags[j].load_data_from_agents(ags_rest)

                # self.ag1.load_data_from_agents(self.ag2_name)
                # self.ag2.load_data_from_agents(self.ag1_name)


            for j in range(len(ags)):
                other_agents = ags[:j] + ags[j+1:]
                ags[j].run(step=i, pre_share=pre_share, share=share, other_agents=other_agents)
            # self.ag1.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag2)
            # self.ag2.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag1)

            if share:
                # clear data
                for ag in ags:
                    ag.clear_agent_data()
                # self.ag1.clear_agent_data()
                # self.ag2.clear_agent_data()

            t2 = time.time()
            print("Time consumed: ", t2 - t1)

    def run_1_agent(self, ag):
        for i in range(NUM_STEPS):
            ag.sample()
            ag.run(step=i)

    def run_simulator(self, replicates=1):
        self.result_taichi = SRC("TAICHI")
        self.result_monk = SRC("Monk")
        self.result_three_body = SRC("TRHEE_BODY")

        for i in range(replicates):
            print("replicate: ", i)
            blockPrint()
            t1 = time.time()

            self.ag1_name = "S1"
            self.ag2_name = "S2"
            self.ag1 = Agent(self.ag1_name, plot=False)
            self.ag1.prepare_run(np.random.randint(len(self.ag1.waypoints)))
            self.ag2 = Agent(self.ag2_name, plot=False)
            self.ag2.prepare_run(np.random.randint(len(self.ag2.waypoints)))
            self.run_multiple_agents([self.ag1, self.ag2])
            self.result_taichi.append(self.ag1)

            self.ag3 = Agent("S3", plot=False)
            self.ag3.prepare_run(np.random.randint(len(self.ag3.waypoints)))
            self.run_1_agent(self.ag3)
            self.result_monk.append(self.ag3)

            # self.ag4 = Agent("TB1", plot=False)
            # self.ag4.prepare_run(np.random.randint(len(self.ag4.waypoints)))
            # self.ag5 = Agent("TB2", plot=False)
            # self.ag5.prepare_run(np.random.randint(len(self.ag5.waypoints)))
            # self.ag6 = Agent("TB3", plot=False)
            # self.ag6.prepare_run(np.random.randint(len(self.ag6.waypoints)))
            # self.run_multiple_agents([self.ag4, self.ag5, self.ag6])
            # self.result_three_body.append(self.ag4)

            enablePrint()
            t2 = time.time()
            print("Time consumed: ", t2 - t1)


    def check_taichi(self):
        a1 = [1000, 0]
        a2 = [500, 600]
        self.update_compass(a1, a2)
        self.get_taichi()
        plt.figure(figsize=(5, 5))

        plt.plot(self.middle_point[1], self.middle_point[0], '.', label="Boat location, WIFI-base station")
        plt.gca().add_patch(self.taichi_circle)
        w0 = Wedge((self.middle_point[1], self.middle_point[0]), self.radius_skp * 2,
                   rad2deg(self.ag2_angle_to_middle_point), rad2deg(self.ag2_angle_to_middle_point) + 180, fc='black', edgecolor='black')

        w1 = Wedge((self.ag1_skp_loc[1], self.ag1_skp_loc[0]), self.radius_skp,
                   rad2deg(self.ag1_angle_to_middle_point), rad2deg(self.ag1_angle_to_middle_point) + 180, fc='black', edgecolor='black')
        w2 = Wedge((self.ag2_skp_loc[1], self.ag2_skp_loc[0]), self.radius_skp,
                   rad2deg(self.ag2_angle_to_middle_point), rad2deg(self.ag2_angle_to_middle_point) + 180, fc='white', edgecolor='black')

        w3 = Wedge((self.middle_point[1], self.middle_point[0]), self.radius_skp * 2,
                   rad2deg(self.ag1_angle_to_middle_point), rad2deg(self.ag1_angle_to_middle_point) + 180, fc='white', edgecolor='white')

        w4 = Wedge((self.ag1_skp_loc[1], self.ag1_skp_loc[0]), LOITER_RADIUS,
                   0, 360, fc='white', edgecolor='black')
        w5 = Wedge((self.ag2_skp_loc[1], self.ag2_skp_loc[0]), LOITER_RADIUS,
                   0, 360, fc='black', edgecolor='white')

        plt.gca().add_artist(w0)
        plt.gca().add_artist(w2)
        plt.gca().add_artist(w3)
        plt.gca().add_artist(w1)
        plt.gca().add_artist(w4)
        plt.gca().add_artist(w5)

        plt.plot(a1[1], a1[0], 'y.')
        plt.plot(a2[1], a2[0], 'b.')

        plt.gca().set_aspect(1)

        plt.show()
        pass

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
    tc.split_regions()
    # tc.check_taichi()
    # tc.run()
    # tc.run_simulator(50)





