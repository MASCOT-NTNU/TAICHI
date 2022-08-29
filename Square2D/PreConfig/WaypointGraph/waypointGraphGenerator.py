"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""
import pandas as pd
from TAICHI.Square2D.Config.Config import FILEPATH, DISTANCE_NEIGHBOUR_WAYPOINT
from TAICHI.Square2D.PreConfig.WaypointGraph.HexagonalGrid2D import HexgonalGrid2DGenerator

polygon_border = FILEPATH + "Config/polygon_border.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = None

grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR_WAYPOINT)

grid_xy = grid.grid_xy

df = pd.DataFrame(grid_xy, columns=['x', 'y'])
df.to_csv(FILEPATH + "Config/WaypointGraph.csv", index=False)

import matplotlib.pyplot as plt
plt.plot(grid_xy[:, 1], grid_xy[:, 0], 'k.')
plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
plt.show()


