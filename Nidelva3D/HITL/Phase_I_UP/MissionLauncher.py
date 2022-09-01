""" Mission Launcher

This module launches the mission for the adaptive sampling mission TAICHI.

Example:

Attributes:
    counter_waypoint_adaptive:

    counter_waypoint_nonadaptive:

    counter_waypoint_data_assimilation:

Methods:
    run():



"""


import rospy




class MissionLauncher:

    def __init__(self):
        """ (MissionLauncher) -> NoneType

        Initialise a mission.


        """


    def run(self):
        """ (MissionLauncher) -> NoneType

        Runs the mission.


        """

        # set waypoint (lat, lon, depth, speed)

        while not rospy.is_shutdown():
            pass



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    m = MissionLauncher()
