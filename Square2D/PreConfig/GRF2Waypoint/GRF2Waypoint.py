"""
This script generates hash table for grf & waypoint
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""


from TAICHI.Square2D.Config.Config import FILEPATH, FIGPATH
from usr_func import *
import pickle


class HashGRF2Waypoint:

    def __init__(self):
        self.load_waypoint()
        self.load_grf_grid()
        self.get_gmrf2waypoint_hash_tables()
        self.get_waypoint2grf_hash_tables()

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv")

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH + "Config/GRFGrid.csv")

    def get_gmrf2waypoint_hash_tables(self):
        self.grf2waypoint_hash_table = dict()

        t1 = time.time()
        for i in range(self.grf_grid.shape[0]):
            x = self.grf_grid['x'][i]
            y = self.grf_grid['y'][i]
            ind = get_ind_at_location2d_xy(self.waypoints.to_numpy(), x, y)
            self.grf2waypoint_hash_table[i] = ind
        t2 = time.time()

        filehandler = open(FILEPATH+'Config/HashGRF2Waypoint.p', 'wb')
        with filehandler as f:
            pickle.dump(self.grf2waypoint_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def get_waypoint2grf_hash_tables(self):
        self.waypoint2grf_hash_table = dict()

        t1 = time.time()
        for i in range(self.waypoints.shape[0]):
            x = self.waypoints['x'][i]
            y = self.waypoints['y'][i]
            ind = get_ind_at_location2d_xy(self.grf_grid.to_numpy(), x, y)
            self.waypoint2grf_hash_table[i] = ind
        t2 = time.time()

        filehandler = open(FILEPATH+'Config/HashWaypoint2GRF.p', 'wb')
        with filehandler as f:
            pickle.dump(self.waypoint2grf_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def check_hash(self):
        self.error_grf2waypoint = []
        self.error_waypoint2grf = []
        for i in range(1000):
            ind = np.random.randint(self.waypoints.shape[0])
            loc_waypoint = self.waypoints.to_numpy()[ind].flatten()
            loc_gmrf = self.grf_grid.to_numpy()[self.waypoint2grf_hash_table[ind]].flatten()
            error_waypoint2gmrf = self.get_error(loc_waypoint, loc_gmrf)
            self.error_waypoint2grf.append(error_waypoint2gmrf)

            ind = np.random.randint(self.grf_grid.shape[0])
            loc_gmrf = self.grf_grid.to_numpy()[ind].flatten()
            loc_waypoint = self.waypoints.to_numpy()[self.grf2waypoint_hash_table[ind]].flatten()
            error_gmrf2waypoint = self.get_error(loc_waypoint, loc_gmrf)
            self.error_grf2waypoint.append(error_gmrf2waypoint)

    def get_error(self, loc1, loc2):
        error = np.sqrt((loc1[0] - loc2[0]) ** 2 +
                 (loc1[1] - loc2[1]) ** 2)
        return error

    def plot_mataching(self):
        # ind = np.random.randint(0, self.waypoints.shape[0], 1000)
        ind = np.arange(self.waypoints.shape[0])
        self.waypoints = self.waypoints.to_numpy()
        self.grf_grid = self.grf_grid.to_numpy()

        plt.figure(figsize=(15, 15))
        plt.plot(self.waypoints[ind, 1], self.waypoints[ind, 0], 'ro', label='waypoint')
        for i in ind:
            plt.plot(self.grf_grid[self.waypoint2grf_hash_table[i], 1],
                     self.grf_grid[self.waypoint2grf_hash_table[i], 0], 'gx')
        plt.plot(self.grf_grid[self.waypoint2grf_hash_table[i], 1],
                 self.grf_grid[self.waypoint2grf_hash_table[i], 0], 'gx', label='gmrf')
        plt.legend()
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.title("Waypoint 2 GMRF grid mapping illustration")
        plt.savefig(FIGPATH+"waypoint2grf.jpg")
        plt.show()

    def check_grid(self):
        plt.figure(figsize=(15, 15))
        plt.plot(self.grf_grid.iloc[:, 0], self.grf_grid.iloc[:, 1], 'k.')
        plt.plot(self.waypoints.iloc[:, 0], self.waypoints.iloc[:, 1], 'r*', markersize=10)
        plt.show()
        pass

if __name__ == "__main__":
    n = HashGRF2Waypoint()
    n.check_grid()
    # n.check_hash()
    # n.check_in_3d()
    n.plot_mataching()






