"""
This script generates path for three agents to move towards the mission start location

It uses TBS as the starting location and first waypoint in the lawnmower patten as the end location.
It then generates three parallel paths for each agent

0 - USV
1 - Agent 1 (AUV-Thor)
2 - Agent 2 (AUV-Harald)

Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-07-11
"""

import pandas as pd
import numpy as np
from usr_func import xy2latlon
from TAICHI.Nidelva3D.Config.Config import FILEPATH

LATITUDE_START, LONGITUDE_START = 63.441178, 10.350178  # TBS location
lawnmower = pd.read_csv(FILEPATH + "HITL/Config/lawnmower.csv").to_numpy()
LATITUDE_END, LONGITUDE_END = lawnmower[0, :2]
SAFETY_DISTANCE = 25  # [m] distance between latitudes


lat_start_1, lon_start_1 = xy2latlon(SAFETY_DISTANCE, 0, LATITUDE_START, LONGITUDE_START)
lat_end_1, lon_end_1 = xy2latlon(SAFETY_DISTANCE, 0, LATITUDE_END, LONGITUDE_END)


lat_start_2, lon_start_2 = xy2latlon(-SAFETY_DISTANCE, 0, LATITUDE_START, LONGITUDE_START)
lat_end_2, lon_end_2 = xy2latlon(-SAFETY_DISTANCE, 0, LATITUDE_END, LONGITUDE_END)

# Agent 1
df = pd.DataFrame(np.array([[lat_start_1, lon_start_1],
                            [lat_end_1, lon_end_1]]), columns=['lat', 'lon'])
df.to_csv(FILEPATH + "HITL/Config/pre_path_1.csv", index=False)

# Agent 2
df = pd.DataFrame(np.array([[lat_start_2, lon_start_2],
                            [lat_end_2, lon_end_2]]), columns=['lat', 'lon'])
df.to_csv(FILEPATH + "HITL/Config/pre_path_2.csv", index=False)

# USV
df = pd.DataFrame(np.array([[LATITUDE_START, LONGITUDE_START],
                            [LATITUDE_END, LONGITUDE_END]]), columns=['lat', 'lon'])
df.to_csv(FILEPATH + "HITL/Config/pre_path_0.csv", index=False)

print("Paths are saved successfully!")


