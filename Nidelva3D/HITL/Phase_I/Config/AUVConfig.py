#!/usr/bin/env python3

import rospy
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms

# == Which depth CTD data needs to be discard
MIN_DEPTH_FOR_DATA_ASSIMILATION = .25
DEPTH_TOLERANCE = .5
# ==

# == Speed
SPEED = 1.5 # speed of AUV
# ==

# == Adaframe
WAYPOINT_UPDATE_TIME = 5.
# ==

# == YoYo
YOYO_LATERAL_DISTANCE = 60.
YOYO_VERTICAL_DISTANCE = 5.
# ==

# == Depth
DEPTH_TOP = .5
DEPTH_BOTTOM = 5.5
# ==

