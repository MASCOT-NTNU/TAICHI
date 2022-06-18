"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""

# == random seed
import os

# == GP kernel
SIGMA = .5
LATERAL_RANGE = 1
NUGGET = .01
THRESHOLD = .7
# ==

# == Grid
XLIM = [0, 1]
YLIM = [0, 1]
LATITUDE_ORIGIN = 0
LONGITUDE_ORIGIN = 0
DISTANCE_NEIGHBOUR = .025
# ==

# == Waypoint
DISTANCE_NEIGHBOUR_WAYPOINT = .05
# ==

# == Path planner
NUM_STEPS = 80
# ==

# == Directories
FILEPATH = os.getcwd() + "/TAICHI/Square2D/"
FIGPATH = FILEPATH + "fig/"
PATH_REPLICATES = FIGPATH + "Replicates/"
# ==

# == Plotting
from matplotlib.cm import get_cmap
CMAP = get_cmap("BrBG", 10)
# ==

# == TAICHI
LOITER_RADIUS = 15 / 1000  # [m], radius used for loitering
SAFETY_DISTANCE = 50 / 1000  # [m]

