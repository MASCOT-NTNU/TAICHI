"""
This script simulates TAICHI's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""
import matplotlib.pyplot as plt

from usr_func import *
from Config.Config import *
from Agent import Agent
# from SimulationResultContainer import SimulationResultContainer as SRC
from spde import spde


class TAICHI:

    def __init__(self):
        print("Hello, this is TAICHI")

    def get_middle_location(self, loc1=None, loc2=None):
        middle_point = [(loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2]
        return middle_point

    def get_station_keeping_loc(self, loc=None, middle_loc=None):
        alpha = np.math.atan2(loc[0] - middle_loc[0],
                              loc[1] - middle_loc[1])
        x = middle_loc[0] + RADIUS_STATION_KEEPING * np.sin(alpha)
        y = middle_loc[1] + RADIUS_STATION_KEEPING * np.cos(alpha)
        return [x, y]

    def get_legal_indices(self, waypoints=None, loc=None, middle_loc=None):
        dx = waypoints[:, 0] - middle_loc[0]
        dy = waypoints[:, 1] - middle_loc[1]

        vec_waypoint_to_middle = np.vstack((dx, dy)) .T
        vec_middle_to_loc = vectorise([loc[0] - middle_loc[0], loc[1] - middle_loc[1]])
        vec_prod = vec_waypoint_to_middle @ vec_middle_to_loc

        same_side = (vec_prod > 0).flatten()
        ind_legal = np.arange(waypoints.shape[0])[same_side]  # get indices for the same side as agent 1
        return ind_legal

    def run_twin_agents(self, ag1, ag2):
        waypoints = ag1.waypoints

        ag1_loc = waypoints[ag1.ind_current_waypoint]  # use next waypoint since system requires
        ag2_loc = waypoints[ag2.ind_current_waypoint]  # pre-advanced calculation

        middle_loc = self.get_middle_location(ag1_loc, ag2_loc)
        ag1_ind_legal = self.get_legal_indices(waypoints, ag1_loc, middle_loc)
        ag2_ind_legal = self.get_legal_indices(waypoints, ag2_loc, middle_loc)

        for i in range(NUM_STEPS):
            print("Step: ", i)
            share = False
            pre_share = False

            t1 = time.time()
            ag1.sample()                                     # step 1
            ag2.sample()

            if (i + 2) % DATA_SHARING_GAP == 0:
                pre_share = True
                ag1_loc = waypoints[ag1.ind_next_waypoint]  # use next waypoint since system requires
                ag2_loc = waypoints[ag2.ind_next_waypoint]  # pre-advanced calculation

                middle_loc = self.get_middle_location(ag1_loc, ag2_loc)
                ag1_sk_loc = self.get_station_keeping_loc(ag1_loc, middle_loc)
                ag2_sk_loc = self.get_station_keeping_loc(ag2_loc, middle_loc)

                # update agent ind_pioneer waypoint to taichi position
                ag1.update_pioneer_waypoint(waypoint_location=ag1_sk_loc)
                ag2.update_pioneer_waypoint(waypoint_location=ag2_sk_loc)

                ag1_ind_legal = self.get_legal_indices(waypoints, ag1_sk_loc, middle_loc)
                ag2_ind_legal = self.get_legal_indices(waypoints, ag2_sk_loc, middle_loc)

            elif i > 0 and i % DATA_SHARING_GAP == 0:
                share = True
                # save data from agent1, agent2
                ag1.save_agent_data()                        # step 2
                ag2.save_agent_data()
                # load data from agent1, agent2
                ag1.load_data_from_other_agent(ag2)
                ag2.load_data_from_other_agent(ag1)

            ag1.run(step=i, pre_share=pre_share, share=share, other_agent=ag2, ind_legal=ag1_ind_legal)  # step 4
            ag2.run(step=i, pre_share=pre_share, share=share, other_agent=ag1, ind_legal=ag2_ind_legal)  # step 4

            if share:
                ag1.clear_agent_data()
                ag2.clear_agent_data()

            t2 = time.time()
            print("One step running takes: ", t2 - t1)

    def run_simulator(self, replicates=1):
        self.result_taichi = SRC("TAICHI")
        self.result_monk = SRC("Monk")
        # self.result_three_body = SRC("TRHEE_BODY")
        ag1_loc = [63.451022, 10.396262, .5]
        ag2_loc = [63.452381, 10.424680, .5]
        ag3_loc = [63.451022, 10.396262, .5]

        waypoints = self.load_waypoints()

        for i in range(replicates):
            print("replicate: ", i)
            t_start = time.time()

            blockPrint()

            self.generate_simulated_truth()

            ag1 = Agent("TAICHI_YIN", plot=False)
            ag2 = Agent("TAICHI_YANG", plot=False)
            ag3 = Agent("MONK", plot=False)
            ag1.prepare_run(ag1_loc)
            ag2.prepare_run(ag2_loc)
            ag3.prepare_run(ag3_loc)

            ag1_loc = waypoints[ag1.ind_current_waypoint]  # use next waypoint since system requires
            ag2_loc = waypoints[ag2.ind_current_waypoint]  # pre-advanced calculation
            ag3_loc = waypoints[ag3.ind_current_waypoint]  # pre-advanced calculation

            middle_loc = self.get_middle_location(ag1_loc, ag2_loc)
            ag1_ind_legal = self.get_legal_indices(waypoints, ag1_loc, middle_loc)
            ag2_ind_legal = self.get_legal_indices(waypoints, ag2_loc, middle_loc)

            enablePrint()

            for i in range(NUM_STEPS):

                print("Step: ", i)

                blockPrint()

                share = False
                pre_share = False

                t1 = time.time()
                ag1.sample()  # step 1
                ag2.sample()
                ag3.sample()

                if (i + 2) % DATA_SHARING_GAP == 0:
                    pre_share = True
                    ag1_loc = waypoints[ag1.ind_next_waypoint]  # use next waypoint since system requires
                    ag2_loc = waypoints[ag2.ind_next_waypoint]  # pre-advanced calculation

                    middle_loc = self.get_middle_location(ag1_loc, ag2_loc)
                    ag1_sk_loc = self.get_station_keeping_loc(ag1_loc, middle_loc)
                    ag2_sk_loc = self.get_station_keeping_loc(ag2_loc, middle_loc)

                    # update agent ind_pioneer waypoint to taichi position
                    ag1.update_pioneer_waypoint(waypoint_location=ag1_sk_loc)
                    ag2.update_pioneer_waypoint(waypoint_location=ag2_sk_loc)

                    ag1_ind_legal = self.get_legal_indices(waypoints, ag1_sk_loc, middle_loc)
                    ag2_ind_legal = self.get_legal_indices(waypoints, ag2_sk_loc, middle_loc)

                elif i > 0 and i % DATA_SHARING_GAP == 0:
                    share = True
                    # save data from agent1, agent2
                    ag1.save_agent_data()  # step 2
                    ag2.save_agent_data()
                    # load data from agent1, agent2
                    ag1.load_data_from_other_agent(ag2)
                    ag2.load_data_from_other_agent(ag1)

                ag1.run(step=i, pre_share=pre_share, share=share, other_agent=ag2, ind_legal=ag1_ind_legal)  # step 4
                ag2.run(step=i, pre_share=pre_share, share=share, other_agent=ag1, ind_legal=ag2_ind_legal)  # step 4
                ag3.run(step=i)

                if share:
                    ag1.clear_agent_data()
                    ag2.clear_agent_data()

                t2 = time.time()

                enablePrint()

                print("One step running takes: ", t2 - t1)

            self.result_taichi.append(ag1)
            self.result_monk.append(ag3)

            # self.ag4 = Agent("TB1", plot=False)
            # self.ag4.prepare_run(np.random.randint(len(self.ag4.waypoints)))
            # self.ag5 = Agent("TB2", plot=False)
            # self.ag5.prepare_run(np.random.randint(len(self.ag5.waypoints)))
            # self.ag6 = Agent("TB3", plot=False)
            # self.ag6.prepare_run(np.random.randint(len(self.ag6.waypoints)))
            # self.run_multiple_agents([self.ag4, self.ag5, self.ag6])
            # self.result_three_body.append(self.ag4)
            t_end = time.time()
            print("One replicate takes: ", t_end - t_start)
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
        print("Simulation result is saved successfully!")

    def load_waypoints(self):
        waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("Waypoint is loaded successfully!")
        return waypoints

    def generate_simulated_truth(self):
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        path_mu_truth = FILEPATH + "Config/Data/data_mu.csv"
        weight = np.random.uniform(.5, 1)
        self.simulated_truth = (weight * pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1) +
                                (1 - weight) * vectorise(self.gmrf_model.mu))  #TODO: JUST for simulation, to remove!!!
        df = pd.DataFrame(self.simulated_truth)
        df.to_csv(FILEPATH + "Config/Data/data_mu_truth.csv", index=False)
        print("truth is generated successfully!")

    def check_taichi(self):
        ag1_loc = [63.451022, 10.396262, .5]
        ag2_loc = [63.452381, 10.424680, .5]
        self.ag1 = Agent("T1", plot=True)
        self.ag2 = Agent("T2", plot=True)
        self.ag1.prepare_run(ag1_loc)
        self.ag2.prepare_run(ag2_loc)
        self.run_twin_agents(self.ag1, self.ag2)

    def check_legal_indices(self):
        ag1_loc = [63.461022, 10.406262, .5]
        ag2_loc = [63.442381, 10.414680, .5]

        self.ag1 = Agent("T1")
        self.ag2 = Agent("T2")
        self.ag1.prepare_run(ag1_loc)
        self.ag2.prepare_run(ag2_loc)

        waypoints = self.ag1.waypoints
        self.l1 = self.l2latlon(ag1_loc)
        self.l2 = self.l2latlon(ag2_loc)

        middle_loc = self.get_middle_location(self.l1, self.l2)
        self.il_l1 = self.get_legal_indices(waypoints, self.l1, middle_loc)
        self.il_l2 = self.get_legal_indices(waypoints, self.l2, middle_loc)

        self.l1_sk = self.get_station_keeping_loc(self.l1, middle_loc)
        self.l2_sk = self.get_station_keeping_loc(self.l2, middle_loc)

        plt.plot(waypoints[:, 1], waypoints[:, 0], 'k.', alpha=.01)
        plt.plot(self.l1[1], self.l1[0], 'r.', markersize=20, alpha=.5, label="OldA1")
        plt.plot(self.l2[1], self.l2[0], 'b.', markersize=20, alpha=.5, label="OldA2")

        plt.plot(self.l1_sk[1], self.l1_sk[0], 'r.', markersize=20, label="A1")
        plt.plot(self.l2_sk[1], self.l2_sk[0], 'b.', markersize=20, label="A2")

        plt.plot(waypoints[self.il_l1, 1], waypoints[self.il_l1, 0], "r*", alpha=.3)
        plt.plot(waypoints[self.il_l2, 1], waypoints[self.il_l2, 0], "b*", alpha=.3)

        plt.plot(middle_loc[1], middle_loc[0], 'c*', markersize=20)
        plt.show()

    def l2latlon(self, loc):
        lat, lon, depth = loc
        x, y = latlon2xy(lat, lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        return [x, y, depth]

    def l2xy(self, loc):
        x, y, z = loc
        lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        return [lat, lon, z]


if __name__ == "__main__":
    tc = TAICHI()
    tc.run_simulator(50)
    # tc.check_taichi()
    # tc.check_legal_indices()
    # tc.check_taichi()
    # tc.run()
    # tc.run_simulator(50)




