"""
This script simulates TAICHI's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""

from usr_func import *
from TAICHI.Simulation.Config.Config import *
from TAICHI.Simulation.Agent import Agent


class TAICHI:

    def __init__(self):
        self.center_of_universe = [0, 0]
        self.radius_of_universe = LOITER_RADIUS + SAFETY_DISTANCE
        # self.setup_agents()
        print("Hello, this is TAICHI")

    def setup_agents(self):
        self.ag1_name = "A1"
        self.ag2_name = "A2"
        self.ag1 = Agent(self.ag1_name)
        self.ag1.set_starting_location(AGENT1_START_LOCATION)
        self.ag1.prepare_run()
        self.ag2 = Agent(self.ag2_name)
        self.ag2.set_starting_location(AGENT2_START_LOCATION)
        self.ag2.prepare_run()

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

        self.taichi_circle = Circle((self.center_of_universe[1], self.center_of_universe[0]),
                                    radius=self.radius_of_universe*2, fill=False, edgecolor='k')
        # self.taichi_circle = Ellipse((self.center_of_universe[1], self.center_of_universe[0]),
        #                              width=self.radius_of_universe*2, height=self.radius_of_universe*2,
        #                              fill=False, edgecolor='r')

    def get_taichi(self):
        self.agent1_new_location = [self.center_of_universe[0] + self.radius_of_universe * np.sin(self.angle1),
                                    self.center_of_universe[1] + self.radius_of_universe * np.cos(self.angle1)]
        self.agent2_new_location = [self.center_of_universe[0] + self.radius_of_universe * np.sin(self.angle2),
                                    self.center_of_universe[1] + self.radius_of_universe * np.cos(self.angle2)]

    def run(self):
        for i in range(NUM_STEPS):
            print("Step: ", i)
            share = False
            pre_share = False

            t1 = time.time()

            self.ag1.sample()
            self.ag2.sample()

            if (i + 2) % DATA_SHARING_GAP == 0:
                print("only pre share")
                pre_share = True

                ag1_loc = self.ag1.waypoints[self.ag1.ind_next_waypoint]  # use next waypoint since system requires
                ag2_loc = self.ag2.waypoints[self.ag2.ind_next_waypoint]  # pre-advanced calculation
                self.update_universe(ag1_loc, ag2_loc)
                self.get_taichi()

                # update agent ind_pioneer waypoint to taichi position
                self.ag1.update_pioneer_waypoint(waypoint_location=self.agent1_new_location)
                self.ag2.update_pioneer_waypoint(waypoint_location=self.agent2_new_location)

            elif i > 0 and i % DATA_SHARING_GAP == 0:
                print("now share")
                share = True

                # save data from agent1, agent2
                self.ag1.save_agent_data()
                self.ag2.save_agent_data()

                # load data from agent1, agent2
                self.ag1.load_data_from_agent(self.ag2_name)
                self.ag2.load_data_from_agent(self.ag1_name)

            self.ag1.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag2)
            self.ag2.run(step=i, pre_share=pre_share, share=share, another_agent=self.ag1)

            if share:
                # clear data
                self.ag1.clear_agent_data()
                self.ag2.clear_agent_data()

            t2 = time.time()
            print("Time consumed: ", t2 - t1)

    def check_taichi(self):
        a1 = [250, 250]
        a2 = [500, 600]
        self.update_universe(a1, a2)
        self.get_taichi()
        plt.figure(figsize=(5, 5))

        plt.plot(self.center_of_universe[1], self.center_of_universe[0], '.', labal='Boat location, WIFI-base station')
        plt.gca().add_patch(self.taichi_circle)
        w0 = Wedge((self.center_of_universe[1], self.center_of_universe[0]), self.radius_of_universe*2,
                   rad2deg(self.angle2), rad2deg(self.angle2) + 180, fc='black', edgecolor='black')

        w1 = Wedge((self.agent1_new_location[1], self.agent1_new_location[0]), self.radius_of_universe,
                   rad2deg(self.angle1), rad2deg(self.angle1) + 180, fc='black', edgecolor='black')
        w2 = Wedge((self.agent2_new_location[1], self.agent2_new_location[0]), self.radius_of_universe,
                   rad2deg(self.angle2), rad2deg(self.angle2) + 180, fc='white', edgecolor='black')

        w3 = Wedge((self.center_of_universe[1], self.center_of_universe[0]), self.radius_of_universe*2,
                   rad2deg(self.angle1), rad2deg(self.angle1) + 180, fc='white', edgecolor='white')

        w4 = Wedge((self.agent1_new_location[1], self.agent1_new_location[0]), LOITER_RADIUS,
                   0, 360, fc='white', edgecolor='black')
        w5 = Wedge((self.agent2_new_location[1], self.agent2_new_location[0]), LOITER_RADIUS,
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

if __name__ == "__main__":
    tc = TAICHI()
    tc.check_taichi()
    # tc.run()


