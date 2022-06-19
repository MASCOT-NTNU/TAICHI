"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-13
"""
import os
import numpy as np
from usr_func import latlon2xy

# == Sys
working_directory = os.getcwd()
FILEPATH = working_directory + "/TAICHI/Nidelva3D/"
FIGPATH = FILEPATH + "fig/"
# ==

# == GP kernel
THRESHOLD = 27
# ==

# == GMRF
GMRF_DISTANCE_NEIGHBOUR = 32
# ==

# == Path planner
NUM_STEPS = 35
# ==

# == Boundary box
BOX = np.load(FILEPATH+"models/grid.npy")
LAT_BOX = BOX[:, 2]
LON_BOX = BOX[:, -1]
LATITUDE_ORIGIN = LAT_BOX[0]
LONGITUDE_ORIGIN = LON_BOX[0]
xbox, ybox = latlon2xy(LAT_BOX, LON_BOX, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
ROTATED_ANGLE = np.math.atan2(xbox[1] - xbox[0], ybox[1] - ybox[0])
# ==

# == TAICHI setup
LOITER_RADIUS = 15  # [m], radius for the loiter
SAFETY_DISTANCE = 50  # [m], safety distance from the AUV
AGENT1_START_LOCATION = [63.451841, 10.393658, .5]
AGENT2_START_LOCATION = [63.454600, 10.426336, .5]
DATA_SHARING_GAP = 7
# ==



