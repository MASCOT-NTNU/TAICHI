"""
This script creates a hash table to save neighbouring location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""

import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from TAICHI.Square2D.Config.Config import FILEPATH, DISTANCE_NEIGHBOUR_WAYPOINT


class HashNeighbourLocations:

    def __init__(self):
        self.load_coordinates()
        self.get_neighbours_hash_table()

    def load_coordinates(self):
        self.coordinates = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv")

    def get_neighbours_hash_table(self):
        self.neighbour_hash_table = dict()
        self.x_coordinates = self.coordinates['x'].to_numpy()
        self.y_coordinates = self.coordinates['y'].to_numpy()

        t1 = time.time()
        for i in range(self.coordinates.shape[0]):
            delta_x = self.x_coordinates - self.x_coordinates[i]
            delta_y = self.y_coordinates - self.y_coordinates[i]
            self.distance_euclidean = np.sqrt(delta_x ** 2 + delta_y ** 2)
            self.distance_ellipsoid = (delta_x ** 2 / (1.5 * DISTANCE_NEIGHBOUR_WAYPOINT) ** 2) + \
                                      (delta_y ** 2 / (1.5 * DISTANCE_NEIGHBOUR_WAYPOINT) ** 2)
            self.ind_neighbour = np.where((self.distance_ellipsoid <= 1) * (self.distance_euclidean >
                                                                            .1*DISTANCE_NEIGHBOUR_WAYPOINT))[0]
            self.neighbour_hash_table[i] = list(self.ind_neighbour)
        t2 = time.time()

        filehandler = open(FILEPATH+'Config/HashNeighbours.p', 'wb')
        with filehandler as f:
            pickle.dump(self.neighbour_hash_table, f)
        f.close()
        print("Hashing finished, time consumed: ", t2 - t1)

    def plot_neighbours(self, ind):
        # Helix equation
        t = np.linspace(0, 20, 100)
        x = self.y_coordinates
        y = self.x_coordinates

        plt.plot(x, y, 'k.', alpha=.3)
        plt.plot(x[ind], y[ind], 'r.', markersize=20)
        plt.plot(x[self.neighbour_hash_table[ind]], y[self.neighbour_hash_table[ind]], 'g.', markersize=20)
        plt.show()


if __name__ == "__main__":
    n = HashNeighbourLocations()
    n.plot_neighbours(np.random.randint(n.coordinates.shape[0]))
    # n.plot_neighbours(12)







