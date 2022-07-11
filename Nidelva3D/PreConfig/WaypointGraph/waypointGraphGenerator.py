"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-07-11
"""

#% Step I: create wgs coordinates

import pandas as pd
from TAICHI.Nidelva3D.PreConfig.WaypointGraph.HexagonalWaypoint3D import HexgonalGrid3DGenerator
from TAICHI.Nidelva3D.Config.Config import *
from TAICHI.Nidelva3D.PreConfig.WaypointGraph.waypointGraphSetup import DISTANCE_NEIGHBOUR
from usr_func import latlon2xy

polygon_border = FILEPATH + "PreConfig/polygon_border.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = np.empty([10, 2])

depth = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
gridGenerator = HexgonalGrid3DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                                        depth=depth, distance_neighbour=DISTANCE_NEIGHBOUR)
coordinates = gridGenerator.coordinates

df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
df.to_csv(FILEPATH + "Config/WaypointGraph.csv", index=False)

import matplotlib.pyplot as plt
plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.show()

#%% Step III: check with plot
import plotly.graph_objects as go
import numpy as np
import plotly

# Helix equation
t = np.linspace(0, 20, 100)
x = df['y'].to_numpy()
y = df['x'].to_numpy()
z = df['z'].to_numpy()

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        # color=z,                # set color to an array/list of desired values
        # colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FILEPATH+"fig/grid.html", auto_open=True)

#%% Step IV: check with GIS to make sure the grid is produced in the right spot
#TODO: check with GIS after grid discrestisation

#%%
