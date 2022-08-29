"""
AUVSimulator module simulates the data collection behaviour of an AUV in the mission.
It generates all possible data samples along the path from the previous location to the current location.

# Args:
#      __loc: current location at [x, y, z]
#      __loc_prev: previous location at [xp, yp, zp]
#      __salinity: salinity measured at current location.
#      __temperature: temperature measured at current location.
#      __ctd: ctd data in the format of [x, y, z, salinity, temperature]
"""
import numpy as np
from AUVSimulator.CTDSimulator import CTDSimulator
from AUVSimulator.Messenger import Messenger


class AUVSimulator:
    __loc = [0, 0, 0]
    __loc_prev = [0, 0, 0]
    __speed = .0

    def __init__(self):
        self.ctd = CTDSimulator()
        self.messenger = Messenger()

    def move_to_location(self, loc: np.ndarray):
        """
        Move AUV to loc, update previous location to current location.
        Args:
            loc: np.array([x, y, z])
        """
        self.__loc_prev = self.__loc
        self.__loc = loc

    def set_location(self, loc: np.ndarray):
        """
        Set AUV location to loc
        Args:
            loc: np.array([x, y, z])
        """
        self.__loc = loc

    def get_location(self) -> np.ndarray:
        """
        Returns: AUV location
        """
        return self.__loc

    def set_previous_location(self, loc: np.ndarray):
        """
        Set previous AUV location to loc
        Args:
            loc: np.array([x, y, z])
        """
        self.__loc_prev = loc

    def get_previous_location(self) -> np.ndarray:
        """
        Returns: previous AUV location
        """
        return self.__loc_prev

    def set_speed(self, value: float) -> None:
        """
        Set speed for AUV.
        Args:
            value: speed in m/s
        """
        self.__speed = value

    def get_speed(self) -> float:
        return self.__speed

    # def get_ctd_along_path(self, loc: np.ndarray):
    #     """
    #     Append data to ctd according to the measurements from previous location to current location.
    #     """
    #     self.__loc = loc
    #     N = 20
    #     x_start, y_start, z_start = self.__loc_prev
    #     x_end, y_end, z_end = self.__loc
    #     x_path = np.linspace(x_start, x_end, N)
    #     y_path = np.linspace(y_start, y_end, N)
    #     z_path = np.linspace(z_start, z_end, N)
    #     # dataset = np.vstack((x_path, y_path, z_path, np.zeros_like(z_path))).T
    #     # ind, value = self.assimilate_data(dataset)


if __name__ == "__main__":
    s = AUVSimulator()
