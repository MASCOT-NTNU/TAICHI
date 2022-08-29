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
    __loc = np.array([0, 0, 0])
    __loc_prev = np.array([0, 0, 0])
    __speed = 1.2
    __ctd_data = np.empty([0, 4])

    def __init__(self):
        self.ctd = CTDSimulator()
        self.messenger = Messenger()

    def move_to_location(self, loc: np.ndarray):
        """
        Move AUV to loc, update previous location to current location.
        Args:
            loc: np.array([x, y, z])
        """
        self.__set_previous_location(self.__loc)
        self.__set_location(loc)
        self.__collect_data()

    def __set_location(self, loc: np.ndarray):
        """
        Set AUV location to loc
        Args:
            loc: np.array([x, y, z])
        """
        self.__loc = loc

    def __set_previous_location(self, loc: np.ndarray):
        """
        Set previous AUV location to loc
        Args:
            loc: np.array([x, y, z])
        """
        self.__loc_prev = loc

    def set_speed(self, value: float) -> None:
        """
        Set speed for AUV.
        Args:
            value: speed in m/s
        """
        self.__speed = value

    def get_location(self) -> np.ndarray:
        """
        Returns: AUV location
        """
        return self.__loc

    def get_previous_location(self) -> np.ndarray:
        """
        Returns: previous AUV location
        """
        return self.__loc_prev

    def get_speed(self) -> float:
        """
        Returns: AUV speed in m/s
        """
        return self.__speed

    def get_ctd_data(self) -> np.ndarray:
        """
        Returns: CTD dataset
        """
        return self.__ctd_data

    def __collect_data(self) -> None:
        """
        Append data along the trajectory from the previous location to the current location.
        Returns:
        """
        x_start, y_start, z_start = self.__loc_prev
        x_end, y_end, z_end = self.__loc
        dx = x_end - x_start
        dy = y_end - y_start
        dz = z_end - z_start
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        N = int(np.ceil(dist / self.__speed) * 2)
        x_path = np.linspace(x_start, x_end, N)
        y_path = np.linspace(y_start, y_end, N)
        z_path = np.linspace(z_start, z_end, N)
        loc = np.stack((x_path, y_path, z_path), axis=1)
        sal = self.ctd.get_salinity_at_loc(loc)
        self.__ctd_data = np.stack((x_path, y_path, z_path, sal), axis=1)


if __name__ == "__main__":
    s = AUVSimulator()
