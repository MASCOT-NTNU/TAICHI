"""
Planner selects candidate locations and then decide which one to go next.

Args:
    _id_prev: previous index
    _id_now: current index
    _id_next: next index
    _id_pion: pioneer index

"""

from abc import abstractmethod


class Planner:

    __id_prev = 0
    __id_now = 0
    __id_next = 0
    __id_pion = 0

    def __init__(self) -> None:
        """ Initialises the planner. """


    def update_planner(self, id_pioneer: int) -> None:
        """
        Update the planner indices by shifting all the remaining indices.

        Args:
            id_pioneer: index for the pioneer waypoint.
        """
        self.__id_prev = self.__id_now
        self.__id_now = self.__id_next
        self.__id_next = self.__id_pion
        self.__id_pion = id_pioneer

    @abstractmethod
    def get_candidates(self):
        pass

    @abstractmethod
    def get_next_waypoint(self):
        pass

    def get_next_index(self) -> int:
        return self.__id_next

    def get_current_index(self) -> int:
        return self.__id_now

    def get_previous_index(self) -> int:
        return self.__id_prev

    def get_pioneer_index(self) -> int:
        return self.__id_pion

    def set_next_index(self, ind: int) -> None:
        self.__id_next = ind

    def set_current_index(self, ind: int) -> None:
        self.__id_now = ind

    def set_previous_index(self, ind: int) -> None:
        self.__id_prev = ind

    def set_pioneer_index(self, ind: int) -> None:
        self.__id_pion = ind

