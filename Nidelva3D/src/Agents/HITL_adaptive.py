"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-06-18

The goal of the agent is to conduct the autonomous sampling operation by using the following procedure:
- Sense
- Plan
- Act

Sense refers to the in-situ measurements. Once the agent obtains the sampled values in the field. Then it can plan based
on the updated knowledge for the field. Therefore, it can act according to the planned manoeuvres.
"""

from Planner.Myopic3D import Myopic3D
from math import radians
from AUV.AUV import AUV
from WGS import WGS
import numpy as np
import time
import os
import math
import rospy


class Agent:

    __loc_start_wgs = np.array([63.446308, 10.409656, 0.5])
    x, y = WGS.latlon2xy(__loc_start_wgs[0], __loc_start_wgs[1])
    __loc_start = np.array([x, y, 0.5])

    __NUM_STEP = 50
    __counter = 0

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner.
        self.myopic = Myopic3D(kernel="GMRF")

        # s2: setup AUV.
        self.auv = AUV()

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # c1: start the operation from scratch.

        id_start = self.myopic.waypoint_graph.get_ind_from_waypoint(self.__loc_start)
        id_curr = id_start

        # s1: setup the planner -> only once

        id_cand = self.myopic.waypoint_graph.get_ind_neighbours(id_curr)
        id_next = id_cand[np.random.randint(0, len(id_cand))]
        id_cand = self.myopic.waypoint_graph.get_ind_neighbours(id_next)
        id_pion = id_cand[np.random.randint(0, len(id_cand))]

        self.myopic.set_current_index(id_curr)
        self.myopic.set_next_index(id_next)
        self.myopic.set_pioneer_index(id_pion)

        """ Set the AUV parameters """
        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_submerged_time()
        popup_time = self.auv.get_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()
        """ End of setting AUV parameters """

        # a1: move to current location
        wp = self.myopic.waypoint_graph.get_waypoint_from_ind(id_curr)
        lat, lon = WGS.xy2latlon(wp[0], wp[1])
        self.auv.auv_handler.setWaypoint(radians(lat), radians(lon), np.abs(wp[2]), speed=speed)

        t_pop_last = time.time()
        update_time = rospy.get_time()

        ctd_data = []
        while not rospy.is_shutdown():
            if self.auv.init:
                t_now = time.time()
                print("counter: ", self.__counter)

                # s1: append data
                loc_auv = self.auv.get_vehicle_pos()
                ctd_data.append([loc_auv[0], loc_auv[1], loc_auv[2], self.auv.get_salinity()])

                if self.__counter == 0:
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                print("vehicle state: ", self.auv.auv_handler.getState())
                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                    # if self.__counter == 0:
                    #     # s2: get next index using get_pioneer_waypoint
                    #     ind = self.myopic.get_pioneer_waypoint_index()
                    #     self.myopic.set_next_index(ind)
                    #
                    #     # p1: parallel move AUV to the first location
                    #     loc = self.myopic.waypoint_graph.get_waypoint_from_ind(ind)
                    #     lat, lon = WGS.xy2latlon(loc[0], loc[1])
                    #     self.auv.auv_handler.setWaypoint(radians(lat), radians(lon), np.abs(loc[2]), speed=speed)
                    #     update_time = rospy.get_time()
                    #
                    #     # s3: update planner -> so curr and next waypoint is updated
                    #     self.myopic.update_planner()
                    #
                    #     # s4: get pioneer waypoint
                    #     self.myopic.get_pioneer_waypoint_index()
                    #
                    #     # s5: obtain CTD data
                    #     ctd_data = np.array(ctd_data)
                    #
                    #     # s6: assimilate data
                    #     self.myopic.kernel.assimilate_data(ctd_data)
                    #     ctd_data = []
                    # else:
                    ind = self.myopic.get_current_index()
                    loc = self.myopic.waypoint_graph.get_waypoint_from_ind(ind)
                    lat, lon = WGS.xy2latlon(loc[0], loc[1])
                    print("Moving to: ", lat, lon, np.abs(loc[2]))
                    self.auv.auv_handler.setWaypoint(radians(lat), radians(lon), np.abs(loc[2]), speed=speed)
                    update_time = rospy.get_time()

                    # a1: gather AUV data
                    ctd_data = np.array(ctd_data)

                    # a2: update GMRF field
                    t1 = time.time()
                    self.myopic.kernel.assimilate_data(ctd_data)
                    t2 = time.time()
                    print("Time to update GMRF: ", t2 - t1)
                    ctd_data = []

                    # ss2: update planner
                    t1 = time.time()
                    self.myopic.update_planner()
                    t2 = time.time()
                    print("Time to update planner: ", t2 - t1)

                    # ss3: plan ahead.
                    t1 = time.time()
                    self.myopic.get_pioneer_waypoint_index()
                    t2 = time.time()
                    print("Time to plan ahead: ", t2 - t1)

                    if self.__counter >= self.__NUM_STEP:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone,
                                                   iridium_dest=iridium)  # self.ada_state = "surfacing"

                        print("Mission complete! Congrates!")
                        self.auv.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    print("counter: ", self.__counter)
                    print(self.myopic.get_current_index())
                    print(self.myopic.get_trajectory_indices())
                    self.__counter += 1
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def get_counter(self):
        return self.__counter


if __name__ == "__main__":
    a = Agent()
    a.run()


