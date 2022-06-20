"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-20
"""

from usr_func import *
from Config.AUVConfig import * # !!!! ROSPY important
from Config.Config import *
from Knowledge.Knowledge import Knowledge
from AUV import AUV
from spde import spde
import pickle
import concurrent.futures


class Lawnmower:

    def __init__(self):
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.setup_AUV()
        self.update_time = rospy.get_time()
        self.get_lawnmower()
        self.data_agent = np.empty([0, 2])
        print("S1-S10 complete!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("S0: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        print("S1: GMRF model is loaded successfully!")

    def setup_AUV(self):
        self.auv = AUV()
        print("S2: AUV is setup successfully!")

    def get_lawnmower(self):
        self.lawnmower = pd.read_csv(FILEPATH + "Config/lawnmower.csv")
        print("S3: Transect line is setup successfully!")

    def run(self):
        self.auv_data = []
        self.ind_visited_waypoint = []
        self.popup = False

        self.counter_waypoint = 0
        lat_waypoint, lon_waypoint, depth_waypoint = self.lawnmower[self.counter_waypoint, :]
        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint),
                                         depth_waypoint, speed=self.auv.speed)

        t_start = time.time()
        while not rospy.is_shutdown():
            if self.auv.init:
                print("Lawnmower waypoint step: ", self.counter_waypoint+1, " of ", len(self.lawnmower))
                t_end = time.time()
                self.auv_data.append([self.auv.vehicle_pos[0],
                                      self.auv.vehicle_pos[1],
                                      self.auv.vehicle_pos[2],
                                      self.auv.currentSalinity])
                self.auv.current_state = self.auv.auv_handler.getState()
                if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
                        (t_end - t_start) % self.auv.max_submerged_time >= 0):
                    print("Longer than 10 mins, need a long break")
                    self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                           phone_number=self.auv.phone_number,
                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                    t_start = time.time()
                    self.popup = True

                # if self.auv.auv_handler.getState() == "waiting":
                if (self.auv.auv_handler.getState() == "waiting" and
                        rospy.get_time() - self.update_time > WAYPOINT_UPDATE_TIME):
                    print("Arrived the current location")

                    self.counter_waypoint += 1
                    if self.counter_waypoint < len(self.lawnmower):
                        lat_waypoint, lon_waypoint, depth_waypoint = self.lawnmower[
                                                                     self.counter_waypoint, :]
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint),
                                                         depth_waypoint, speed=self.auv.speed)
                    ind_assimilated, salinity_assimilated = self.assimilate_data(np.array(self.auv_data))
                    t1 = time.time()
                    self.gmrf_model.update(rel=salinity_assimilated, ks=ind_assimilated)
                    t2 = time.time()
                    print("Update consumed: ", t2 - t1)
                    if self.counter_waypoint == len(self.lawnmower):
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                               phone_number=self.auv.phone_number,
                                               iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), 0,
                                                         speed=self.auv.speed)
                        print("Mission complete! Congrates!")
                        self.auv.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()
    #
    # def run(self):
    #     self.auv_data = []
    #     self.popup = False
    #
    #     self.counter_waypoint = 0
    #     lat_waypoint, lon_waypoint, depth_waypoint = self.lawnmower[self.counter_waypoint, :]
    #     self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint),
    #                                      depth_waypoint, speed=self.auv.speed)
    #     t_start = time.time()
    #     while not rospy.is_shutdown():
    #         if self.auv.init:
    #             print("Waypoint step: ", self.counter_waypoint+1, " of ", len(self.lawnmower))
    #             t_end = time.time()
    #             self.auv_data.append([self.auv.vehicle_pos[0],
    #                                   self.auv.vehicle_pos[1],
    #                                   self.auv.vehicle_pos[2],
    #                                   self.auv.currentSalinity])
    #             self.auv.current_state = self.auv.auv_handler.getState()
    #             if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
    #                     (t_end - t_start) % self.auv.max_submerged_time >= 0):
    #                 print("Longer than 10 mins, need a long break")
    #                 self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
    #                                        phone_number=self.auv.phone_number,
    #                                        iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
    #                 t_start = time.time()
    #
    #                 ind_assimilated, salinity_assimilated = self.assimilate_data(np.array(self.auv_data))
    #                 self.data_agent = np.append(self.data_agent, np.hstack((vectorise(ind_assimilated),
    #                                                                         vectorise(salinity_assimilated))), axis=0)
    #                 t1 = time.time()
    #                 self.gmrf_model.update(rel=salinity_assimilated, ks=ind_assimilated)
    #                 t2 = time.time()
    #                 print("Update consumed: ", t2 - t1)
    #
    #             self.auv.last_state = self.auv.auv_handler.getState()
    #             self.auv.auv_handler.spin()
    #         self.auv.rate.sleep()

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[-10:, :])
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= MIN_DEPTH_FOR_DATA_ASSIMILATION)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        print("dataset after filtering: ", dataset[-10:, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 1]).T) ** 2
        dz = ((vectorise(dataset[:, 2]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 2]).T) * GMRF_DISTANCE_NEIGHBOUR) ** 2
        dist = dx + dy + dz
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated)


if __name__ == "__main__":
    s = Lawnmower()
    s.run()


