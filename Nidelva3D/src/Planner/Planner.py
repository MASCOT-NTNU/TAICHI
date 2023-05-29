"""
Planner module is the parent class of all the planners. It provides the basic functions for the planner.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-25

Methodology:
    1. It updates the planner indices after each iteration.

Args:
    _id_curr: current index
    _id_next: next index
    _id_pion: pioneer index
"""

from abc import abstractmethod


class Planner:
    """
    Planner module is the parent class of all the planners. It provides the basic functions for the planner.
    """
    def __init__(self) -> None:
        self.__id_curr = 0
        self.__id_next = 0
        self.__id_pion = 0
        self.__traj = []

    def update_planner(self) -> None:
        """
        Update the planner indices by shifting all the remaining indices.
        """
        self.__id_curr = self.__id_next
        self.__id_next = self.__id_pion
        self.__traj.append(self.__id_curr)

    @abstractmethod
    def get_candidates_indices(self):
        pass

    @abstractmethod
    def get_pioneer_waypoint_index(self):
        pass

    def get_next_index(self) -> int:
        return self.__id_next

    def get_current_index(self) -> int:
        return self.__id_curr

    def get_pioneer_index(self) -> int:
        return self.__id_pion

    def get_trajectory_indices(self) -> list:
        return self.__traj

    def set_next_index(self, ind: int) -> None:
        self.__id_next = ind

    def set_current_index(self, ind: int) -> None:
        self.__id_curr = ind

    def set_pioneer_index(self, ind: int) -> None:
        self.__id_pion = ind

