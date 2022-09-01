"""
This script simulates Agent Yin's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""

from usr_func import *
# from TAICHI.Square2D.Config.Config import FILEPATH, FIGPATH, CMAP, THRESHOLD
from TAICHI.Square2D.Config.Config import *
from TAICHI.Square2D.Myopic2D import MyopicPlanning2D
from TAICHI.Square2D.GRF import GRF
from TAICHI.Square2D.PlotFunc import plotf_vector
import pickle
from sklearn.metrics import mean_squared_error
import properscoring


class Agent:

    def __init__(self, name="Agent", plot=False, seed=None):
        self.agent_name = name
        self.plot = plot
        self.load_waypoint()
        self.load_grf_grid()
        self.load_grf_model(seed)
        self.load_hash_neighbours()
        self.load_hash_waypoint2grf()
        self.initialise_function_calls()
        self.setup_data_container()
        print("S1-S5 complete!" + self.agent_name + " is initialised successfully!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH + "Config/GRFGrid.csv").to_numpy()
        self.N_grf_grid = len(self.grf_grid)
        print("S2: GRF grid is loaded successfully!")

    def load_grf_model(self, seed=None):
        self.grf_model = GRF(seed=seed)
        print("S3: GRF model is loaded successfully!")

    def load_hash_neighbours(self):
        neighbour_file = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(neighbour_file)
        neighbour_file.close()
        print("S4: Neighbour hash table is loaded successfully!")

    def load_hash_waypoint2grf(self):
        waypoint2grf_file = open(FILEPATH + "Config/HashWaypoint2GRF.p", 'rb')
        self.hash_waypoint2grf = pickle.load(waypoint2grf_file)
        waypoint2grf_file.close()
        print("S5: Waypoint2GRF hash table is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location2d_xy(self.waypoints, 1, 2)  # used to initialise the function call
        print("S6: Function calls are initialised successfully!")

    def setup_data_container(self):
        self.data_agent = np.empty([0, 2])
        self.ibv = []
        self.uncertainty = []
        self.crps = []
        self.rmse = []
        print("S7: Data container is initialised successfully!")

    def set_starting_location(self, starting_location):
        x, y = starting_location
        self.x_start = x
        self.y_start = y
        print("Starting location is set up successfully!")

    def prepare_run(self, ind_start=None, ind_legal=None):
        if not ind_start:
            self.ind_current_waypoint = get_ind_at_location2d_xy(self.waypoints, self.x_start, self.y_start)
        else:
            self.ind_current_waypoint = ind_start
        print("startingh index: ", self.ind_current_waypoint)
        self.ind_previous_waypoint = self.ind_current_waypoint
        self.ind_next_waypoint = self.ind_current_waypoint
        self.ind_visited_waypoint = []
        self.ind_visited_waypoint.append(self.ind_current_waypoint)

        self.m2_planner = MyopicPlanning2D(grf_model=self.grf_model, waypoint_graph=self.waypoints,
                                           hash_neighbours=self.hash_neighbours, hash_waypoint2grf=self.hash_waypoint2grf,
                                           echo=False)

        if ind_legal is None:
            ind_legal = np.arange(self.waypoints.shape[0])

        self.m2_planner.update_legal_indices(ind_legal)
        self.ind_next_waypoint = self.m2_planner.find_next_waypoint_using_min_eibv(ind_current=self.ind_current_waypoint,
                                                                                   ind_previous=self.ind_previous_waypoint,
                                                                                   ind_visited=self.ind_visited_waypoint)
        print("Ready to run!")

        if self.plot:
            figpath = FIGPATH + self.agent_name + "/"
            title = "GroundTruth"
            filename = figpath + title + ".jpg"
            x = self.grf_grid[:, 0]
            y = self.grf_grid[:, 1]
            self.vmin = np.amin(self.grf_model.mu_truth)
            self.vmax = np.amax(self.grf_model.mu_truth)
            self.stepsize = (self.vmax - self.vmin) / 15

            plt.figure()
            # self.plotf(x, y, self.grf_model.mu_truth, vmin=0, vmax=1, cmap=get_cmap("BrBG", 10), filename=filename)
            plotf_vector(x, y, self.grf_model.mu_truth, title=title, cmap=get_cmap("BrBG", 10), vmin=self.vmin,
                         vmax=self.vmax, stepsize=self.stepsize, cbar_title="Test", threshold=THRESHOLD,
                         xlabel='x', ylabel='y')
            wd = os.path.dirname(filename)
            checkfolder(wd)
            plt.savefig(filename)
            plt.show()

            title = "Prior"
            filename = figpath + title + ".jpg"
            x = self.grf_grid[:, 0]
            y = self.grf_grid[:, 1]
            plt.figure()
            plotf_vector(x, y, self.grf_model.mu_prior, title=title, cmap=get_cmap("BrBG", 10), vmin=self.vmin,
                         vmax=self.vmax, stepsize=self.stepsize, cbar_title="Test", threshold=THRESHOLD,
                         xlabel='x', ylabel='y')
            wd = os.path.dirname(filename)
            checkfolder(wd)
            plt.savefig(filename)
            plt.show()

    def sample(self):
        self.ind_sample_grf = self.get_ind_sample(self.ind_previous_waypoint, self.ind_current_waypoint)
        self.salinity_measured = self.grf_model.mu_truth[self.ind_sample_grf]
        self.data_agent = np.append(self.data_agent, np.hstack((vectorise(self.ind_sample_grf),
                                                                vectorise(self.salinity_measured))),
                                    axis=0)

    def monitor_data(self):
        truth = self.grf_model.mu_truth
        self.rmse.append(mean_squared_error(truth, self.grf_model.mu_cond, squared=True))
        self.uncertainty.append(np.sum(np.diag(self.grf_model.Sigma_cond)))
        self.ibv.append(self.m2_planner.get_eibv_from_grf_model(self.ind_current_waypoint))
        self.crps.append(np.sum(properscoring.crps_gaussian(truth[self.ind_current_waypoint],
                                                            self.grf_model.mu_cond,
                                                            np.diag(self.grf_model.Sigma_cond))))

    def run(self, step=0, pre_share=False, share=False, other_agent=None, ind_legal=None):
        self.monitor_data()
        if share:
            self.ind_measured_by_other_agent = self.data_from_other_agent[:, 0].astype(int)
            self.salinity_measured_from_other_agent = vectorise(self.data_from_other_agent[:, 1])

            self.ind_sample_grf = np.append(self.ind_measured_by_other_agent, self.ind_sample_grf, axis=0)
            self.salinity_measured = np.append(self.salinity_measured_from_other_agent, self.salinity_measured, axis=0)

        t1 = time.time()
        self.grf_model.update_grf_model(ind_measured=self.ind_sample_grf, salinity_measured=self.salinity_measured)
        t2 = time.time()
        print("Update consumed: ", t2 - t1)

        if not pre_share:
            if ind_legal is None:
                ind_legal = np.arange(self.waypoints.shape[0])
            self.m2_planner.update_legal_indices(ind_legal)
            self.ind_next_waypoint = self.m2_planner.find_next_waypoint_using_min_eibv(ind_current=self.ind_current_waypoint,
                                                                                       ind_previous=self.ind_previous_waypoint,
                                                                                       ind_visited=self.ind_visited_waypoint)
        if self.plot:
            figpath = FIGPATH + self.agent_name + "/"
            filename = figpath + "P_{:03d}.jpg".format(step)
            title = "Updated field after {:d} iterations".format(step)

            plt.figure(figsize=(20, 6))
            plt.subplot(131)
            plotf_vector(self.grf_grid[:, 0], self.grf_grid[:, 1], self.grf_model.mu_cond, title=title, vmin=self.vmin,
                         vmax=self.vmax, stepsize=self.stepsize, cmap=get_cmap("BrBG", 10), cbar_title="Test",
                         threshold=THRESHOLD, xlabel='x', ylabel='y')
            x = self.waypoints[:, 0]
            y = self.waypoints[:, 1]
            plt.plot(self.grf_grid[self.ind_sample_grf, 0], self.grf_grid[self.ind_sample_grf, 1], 'c.')
            plt.plot(x[self.ind_visited_waypoint], y[self.ind_visited_waypoint], 'k.-')
            plt.plot(x[self.m2_planner.ind_candidates], y[self.m2_planner.ind_candidates], 'y.')
            plt.plot(x[self.ind_current_waypoint], y[self.ind_current_waypoint], 'r.')
            plt.plot(x[self.ind_next_waypoint], y[self.ind_next_waypoint], 'b*')
            plt.plot(x[self.ind_previous_waypoint], y[self.ind_previous_waypoint], 'g^')

            if other_agent:
                plt.plot(x[other_agent.ind_visited_waypoint], y[other_agent.ind_visited_waypoint], 'k.-', alpha=.3)
                plt.plot(x[other_agent.ind_current_waypoint], y[other_agent.ind_current_waypoint], 'c.', alpha=.3)
            if share:
                x = self.grf_grid[:, 0]
                y = self.grf_grid[:, 1]
                plt.plot(x[self.ind_measured_by_other_agent], y[self.ind_measured_by_other_agent], 'c.', alpha=.3)

            plt.subplot(132)
            plotf_vector(self.grf_grid[:, 0], self.grf_grid[:, 1], np.diag(self.grf_model.Sigma_cond),
                         title="Updated uncertainity field", vmin=0,
                         vmax=.25, cmap=get_cmap("BrBG", 10), cbar_title="Test", xlabel='x', ylabel='y')
            x = self.waypoints[:, 0]
            y = self.waypoints[:, 1]
            plt.plot(self.grf_grid[self.ind_sample_grf, 0], self.grf_grid[self.ind_sample_grf, 1], 'c.')
            plt.plot(x[self.ind_visited_waypoint], y[self.ind_visited_waypoint], 'k.-')
            plt.plot(x[self.m2_planner.ind_candidates], y[self.m2_planner.ind_candidates], 'y.')
            plt.plot(x[self.ind_current_waypoint], y[self.ind_current_waypoint], 'r.')
            plt.plot(x[self.ind_next_waypoint], y[self.ind_next_waypoint], 'b*')
            plt.plot(x[self.ind_previous_waypoint], y[self.ind_previous_waypoint], 'g^')

            if other_agent:
                plt.plot(x[other_agent.ind_visited_waypoint], y[other_agent.ind_visited_waypoint], 'k.-', alpha=.3)
                plt.plot(x[other_agent.ind_current_waypoint], y[other_agent.ind_current_waypoint], 'c.', alpha=.3)
            if share:
                x = self.grf_grid[:, 0]
                y = self.grf_grid[:, 1]
                plt.plot(x[self.ind_measured_by_other_agent], y[self.ind_measured_by_other_agent], 'c.', alpha=.3)

            plt.subplot(133)
            plt.plot(self.waypoints[ind_legal, 0], self.waypoints[ind_legal, 1], 'r*', alpha=.3, markersize=10)
            plt.plot(self.waypoints[ind_legal, 0], self.waypoints[ind_legal, 1], 'b*', alpha=.3, markersize=10)
            plt.plot(self.waypoints[self.ind_current_waypoint, 0],
                     self.waypoints[self.ind_current_waypoint, 1], 'r.', label="AG1 Old")
            plt.plot(self.waypoints[self.ind_visited_waypoint, 0],
                     self.waypoints[self.ind_visited_waypoint, 1], 'k-.')
            plt.xlim([XLIM[0], XLIM[1]])
            plt.ylim([YLIM[0], YLIM[1]])
            if other_agent:
                plt.plot(self.waypoints[other_agent.ind_current_waypoint, 0],
                         self.waypoints[other_agent.ind_current_waypoint, 1], 'b.', label="AG2 Old")

            wd = os.path.dirname(filename)
            checkfolder(wd)
            plt.savefig(filename)
            # plt.show()
            plt.close("all")

        self.ind_previous_waypoint = self.ind_current_waypoint
        self.ind_current_waypoint = self.ind_next_waypoint
        self.ind_visited_waypoint.append(self.ind_current_waypoint)

    def save_agent_data(self):
        datapath = FILEPATH + "AgentsData/" + self.agent_name + ".npy"
        checkfolder(os.path.dirname(datapath))
        np.save(datapath, self.data_agent)
        print("Data from " + self.agent_name + " is saved successfully!")

    def load_data_from_agents(self, ag):
        self.data_from_other_agent = np.empty([0, 2])
        agent_name = ag.agent_name
        datapath = FILEPATH + "AgentsData/" + agent_name + ".npy"
        self.data_from_other_agent = np.append(self.data_from_other_agent, np.load(datapath), axis=0)
        print("Data from " + agent_name + " is loaded successfully!")

    def clear_agent_data(self):
        self.data_agent = np.empty([0, 2])
        print("Data from " + self.agent_name + " is cleared successfully!", self.data_agent)

    def get_ind_sample(self, ind_start, ind_end):
        N = 20
        x_start, y_start = self.waypoints[ind_start, :]
        x_end, y_end = self.waypoints[ind_end, :]
        x_path = np.linspace(x_start, x_end, N)
        y_path = np.linspace(y_start, y_end, N)
        dataset = np.vstack((x_path, y_path, np.zeros_like(x_path))).T
        ind, value = self.assimilate_data(dataset)
        return ind

    def assimilate_data(self, dataset):
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 1]).T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 2])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated)

    def check_agent(self):
        self.set_starting_location([1, 0])
        self.prepare_run()
        for i in range(100):
            print("Step: ", i)
            self.sample()
            self.run(i)
        os.system("say finished")


if __name__ == "__main__":
    # np.random.seed(0)
    a = Agent()
    # a.prepare_run(10)
    a.check_agent()



