"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

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
import numpy as np
import time
import os


class Agent:

    __loc_start = np.array([0, 0, 0])
    __loc_end = np.array([0, 0, 0])
    __NUM_STEP = 50
    __counter = 0

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner.
        self.myopic = Myopic3D()

        # s2: setup AUV simulator.
        self.auv = AUVSimulator()

        # s3: setup Visualiser.
        self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/Myopic3D/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # c1: start the operation from scratch.
        id_start = np.random.randint(0, len(self.myopic.wp.get_waypoints()))
        id_curr = id_start

        # s1: setup the planner -> only once
        self.myopic.set_current_index(id_curr)
        self.myopic.set_next_index(id_curr)

        # a1: move to current location
        self.auv.move_to_location(self.myopic.wp.get_waypoint_from_ind(id_curr))

        t_start = time.time()
        t_pop_last = time.time()

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
                    loc = self.myopic.wp.get_waypoint_from_ind(ind)
                    self.auv.move_to_location(loc)

                    # s3: update planner -> so curr and next waypoint is updated
                    self.myopic.update_planner()

                    # s4: get pioneer waypoint
                    self.myopic.get_pioneer_waypoint_index()

                    # # s5: obtain CTD data
                    ctd_data = self.auv.get_ctd_data()

                    # # s5: assimilate data
                    self.myopic.gmrf.assimilate_data(ctd_data)
                else:
                    ind = self.myopic.get_current_index()
                    loc = self.myopic.wp.get_waypoint_from_ind(ind)
                    self.auv.move_to_location(loc)

                    # a1: gather AUV data
                    ctd_data = self.auv.get_ctd_data()

                    # a2: update GMRF field
                    self.myopic.gmrf.assimilate_data(ctd_data)

                    # ss2: update planner
                    self.myopic.update_planner()

                    # ss3: plan ahead.
                    self.myopic.get_pioneer_waypoint_index()

                    if self.__counter == self.__NUM_STEP:
                        break
                print("counter: ", self.__counter)
                print(self.myopic.get_current_index())
                print(self.myopic.get_trajectory_indices())
                self.visualiser.plot_agent()
                self.__counter += 1

    def get_counter(self):
        return self.__counter


if __name__ == "__main__":
    a = Agent()
    a.run()


