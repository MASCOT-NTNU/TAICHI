"""
Agent zigzag object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

The goal of the agent is to conduct the autonomous sampling operation by using the following procedure:
- Sense
- Plan
- Act

Sense refers to the in-situ measurements. Once the agent obtains the sampled values in the field. Then it can plan based
on the updated knowledge for the field. Therefore, it can act according to the planned manoeuvres.
"""
from Planner.ZigZag import ZigZag
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.Visualiser_zigzag import Visualiser
import numpy as np
import time
import os


class Agent:

    __NUM_STEP = 200
    __counter = 0
    __traj = np.empty([0, 3])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner.
        self.zz = ZigZag()

        # s2: setup AUV simulator.
        self.auv = AUVSimulator()

        # s3: setup Visualiser.
        self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/ZigZag/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # a0: get starting location
        path = self.zz.get_zigzag_path()
        loc_start = path[0]

        # a1: move to current location
        self.auv.move_to_location(loc_start)
        self.__traj = np.append(self.__traj, loc_start.reshape(1, -1))

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

                loc = path[self.__counter]
                self.auv.move_to_location(loc)
                self.__traj = np.append(self.__traj, loc.reshape(1, -1))

                # a1: gather AUV data
                ctd_data = self.auv.get_ctd_data()

                if self.__counter == self.__NUM_STEP:
                    break
                print("counter: ", self.__counter)
                self.visualiser.plot_agent()
                self.__counter += 1

    def get_counter(self):
        return self.__counter

    def get_traj(self):
        return self.__traj


if __name__ == "__main__":
    a = Agent()
    a.run()


