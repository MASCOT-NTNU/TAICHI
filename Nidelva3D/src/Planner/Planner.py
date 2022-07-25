""" Planner object is the superclass for the path planning. It encapsulates essential information for the path planning.

"""

import abc


class Planner:

    _ind_previous = 0
    _ind_current = 0
    _ind_next = 0
    _ind_pioneer = 0

    def __init__(self):
        pass

    def update_planner(self):
        """ (Planner) -> NoneType

        Updates the planner by moving each pointer forward one step.

        """
        self._ind_previous = self._ind_current
        self._ind_current = self._ind_next
        self._ind_next = self._ind_pioneer

    def update_pioneer_ind(self, new_pioneer_ind: int) -> None:
        """ Updates the pioneer index.

        Args:
            new_pioneer_ind: int
        """
        self._ind_pioneer = new_pioneer_ind

    @abc.abstractmethod
    def plan_one_step_ahead(self):
        """ Plans one step ahead.

        """
        raise NotImplementedError("Please use this method properly")


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
