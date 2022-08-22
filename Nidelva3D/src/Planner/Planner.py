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

    _id_prev = 0
    _id_now = 0
    _id_next = 0
    _id_pion = 0

    def __init__(self) -> None:
        """ Initialises the planner. """


    def update_planner(self, id_pioneer: int) -> None:
        """
        Update the planner indices by shifting all the remaining indices.

        Args:
            id_pioneer: index for the pioneer waypoint.
        """
        self._id_prev = self._id_now
        self._id_now = self._id_next
        self._id_next = self._id_pion
        self._id_pion = id_pioneer

    @abstractmethod
    def get_candidates(self):
        pass

    @abstractmethod
    def get_next_waypoint(self):
        pass

    def get_next_index(self) -> int:
        return self._id_next

    def get_current_index(self) -> int:
        return self._id_now

    def get_previous_index(self) -> int:
        return self._id_prev

    def get_pioneer_index(self) -> int:
        return self._id_pion

    def set_next_index(self, ind: int) -> None:
        self._id_next = ind

    def set_current_index(self, ind: int) -> None:
        self._id_now = ind

    def set_previous_index(self, ind: int) -> None:
        self._id_prev = ind

    def set_pioneer_index(self, ind: int) -> None:
        self._id_pion = ind

