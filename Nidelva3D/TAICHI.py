"""
This script simulates TAICHI's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""

from usr_func import *
from TAICHI.Nidelva3D.Config.Config import *
from TAICHI.Nidelva3D.Agent import Agent
from TAICHI.Nidelva3D.SimulationResultContainer import SimulationResultContainer as SRC


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

    def update_universe(self, agent1=None, agent2=None):
        if agent1 is None:
            agent1 = [0, 0]
        if agent2 is None:
            agent2 = [0, 0]
        self.center_of_universe = [(agent1[0] + agent2[0])/2, (agent1[1] + agent2[1])/2]
        self.angle1 = np.math.atan2(agent1[0] - self.center_of_universe[0],
                                    agent1[1] - self.center_of_universe[1])
        self.angle2 = np.math.atan2(agent2[0] - self.center_of_universe[0],
                                    agent2[1] - self.center_of_universe[1])

    def repartition(self):
        self.waypoints = self.ag1.waypoints
        self.update_compass(self.waypoints[self.ag1.ind_current_waypoint],
                            self.waypoints[self.ag2.ind_current_waypoint])

        self.dx = self.waypoints[:, 0] - self.middle_point[0]
        self.dy = self.waypoints[:, 1] - self.middle_point[1]

        vec_middle_to_ag1 = vectorise([self.waypoints[self.ag1.ind_current_waypoint, 0] - self.middle_point[0],
                                       self.waypoints[self.ag1.ind_current_waypoint, 1] - self.middle_point[1]])
        self.vec_prod = np.vstack((self.dx, self.dy)).T @ vec_middle_to_ag1

        self.ind_legal = np.arange(self.waypoints.shape[0])
        self.same_side = (self.vec_prod > 0).flatten()
        self.ind_ag1 = self.ind_legal[self.same_side]  # get indices for the same side as agent 1
        self.ind_ag2 = self.ind_legal[~self.same_side]  # get indices for the opposite side as agent 1
        return self.ind_ag1, self.ind_ag2

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

                self.update_compass(self.waypoints[self.ag1.ind_current_waypoint, :],
                                    self.waypoints[self.ag2.ind_current_waypoint, :])
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

    # def run_multiple_agents(self, ags):
    #
    #     for i in range(NUM_STEPS):
    #         print("Step: ", i)
    #         share = False
    #         pre_share = False
    #
    #         t1 = time.time()
    #
    #         for ag in ags:
    #             ag.sample()
    #         # self.ag1.sample()
    #         # self.ag2.sample()
    #
    #         if (i + 2) % DATA_SHARING_GAP == 0:
    #             # print("only pre share")
    #             pre_share = True
    #
    #             # ag1_loc = self.ag1.waypoints[self.ag1.ind_next_waypoint]  # use next waypoint since system requires
    #             # ag2_loc = self.ag2.waypoints[self.ag2.ind_next_waypoint]  # pre-advanced calculation
    #             # self.update_universe(ag1_loc, ag2_loc)
    #             # self.get_taichi()
    #
    #             # update agent ind_pioneer waypoint to taichi position
    #             # self.ag1.update_pioneer_waypoint(waypoint_location=self.agent1_new_location)
    #             # self.ag2.update_pioneer_waypoint(waypoint_location=self.agent2_new_location)
    #
    #         elif i > 0 and i % DATA_SHARING_GAP == 0:
    #             # print("now share")
    #             share = True
    #
    #             # save data from agent1, agent2
    #             for ag in ags:
    #                 ag.save_agent_data()
    #             # self.ag1.save_agent_data()
    #             # self.ag2.save_agent_data()
    #
    #             # load data from agent1, agent2
    #             for j in range(len(ags)):
    #                 ags_rest = ags[:j] + ags[j+1:]
    #                 ags[j].load_data_from_agents(ags_rest)
    #
    #             # self.ag1.load_data_from_agents(self.ag2_name)
    #             # self.ag2.load_data_from_agents(self.ag1_name)
    #
    #
    #         for j in range(len(ags)):
    #             other_agents = ags[:j] + ags[j+1:]
    #             ags[j].run(step=i, pre_share=pre_share, share=share, other_agents=other_agents)
    #         # self.ag1.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag2)
    #         # self.ag2.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag1)
    #
    #         if share:
    #             # clear data
    #             for ag in ags:
    #                 ag.clear_agent_data()
    #             # self.ag1.clear_agent_data()
    #             # self.ag2.clear_agent_data()
    #
    #         t2 = time.time()
    #         print("Time consumed: ", t2 - t1)

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
        ag1_loc = [63.451022, 10.396262, .5]
        ag2_loc = [63.452381, 10.424680, .5]

        self.setup_agents(ag1_loc, ag2_loc)
        self.run_twin_agents(self.ag1, self.ag2)
        pass


if __name__ == "__main__":
    tc = TAICHI()
    tc.check_taichi()
    # tc.run()
    # tc.run_simulator(50)



