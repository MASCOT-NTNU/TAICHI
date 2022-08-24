"""
Simulator module simulates the data collection behaviour of an AUV in the mission.
It generates all possible data samples along the path from the previous location to the current location.

Args:
     __loc: current location at [x, y, z]
     __loc_prev: previous location at [xp, yp, zp]
     __salinity: salinity measured at current location.
     __temperature: temperature measured at current location.
     __ctd: ctd data in the format of [x, y, z, salinity, temperature
"""
import numpy as np


class Simulator:

    __loc = [0, 0, 0]
    __loc_prev = [0, 0, 0]
    __salinity = .0
    __temperature = .0
    __ctd = [0, 0, 0, 0, 0]

    def __init__(self):
        pass

    def get_location(self):
        return self.__loc

    def get_salinity(self):
        return self.__salinity

    def get_temperature(self):
        return self.__temperature

    def move_to_location(self, loc: np.ndarray):
        """
        Append data to ctd according to the measurements from previous location to current location.
        """
        self.__loc = loc
        N = 20
        x_start, y_start, z_start = self.__loc_prev
        x_end, y_end, z_end = self.__loc
        x_path = np.linspace(x_start, x_end, N)
        y_path = np.linspace(y_start, y_end, N)
        z_path = np.linspace(z_start, z_end, N)
        # dataset = np.vstack((x_path, y_path, z_path, np.zeros_like(z_path))).T
        # ind, value = self.assimilate_data(dataset)
        pass


if __name__ == "__main__":
    s = Simulator()
