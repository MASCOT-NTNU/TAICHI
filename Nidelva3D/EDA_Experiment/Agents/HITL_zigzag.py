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
import math
from Planner.ZigZag import ZigZag
from AUV.AUV2 import AUV
from WGS import WGS
import numpy as np
import time
import os
import rospy


class Agent:

    __NUM_STEP = 0
    __counter = 0

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner.
        self.zz = ZigZag()

        # s2: setup AUV.
        self.auv = AUV()

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # 00: get information from AUV
        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_submerged_time()
        popup_time = self.auv.get_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()

        # a0: get starting location
        path = self.zz.get_zigzag_path()
        loc_start = path[0]
        self.__NUM_STEP = len(path)

        # a1: move to current location
        lat, lon = WGS.xy2latlon(loc_start[0], loc_start[1])
        self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), loc_start[2], speed=speed)

        t_pop_last = time.time()
        update_time = rospy.get_time()

        ctd_data = []
        while not rospy.is_shutdown():
            if self.auv.init:
                t_now = time.time()
                print("counter: ", self.__counter)

                # s1: append data:
                loc_auv = self.auv.get_vehicle_pos()
                ctd_data.append([loc_auv[0], loc_auv[1], loc_auv[2], self.auv.get_salinity()])

                if self.__counter == 0:
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                    loc = path[self.__counter]
                    lat, lon = WGS.xy2latlon(loc[0], loc[1])
                    self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), loc[2], speed=speed)
                    update_time = rospy.get_time()

                    if self.__counter >= self.__NUM_STEP:
                        break
                    print("counter: ", self.__counter)
                    self.__counter += 1

                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def get_counter(self):
        return self.__counter


if __name__ == "__main__":
    a = Agent()
    a.run()


