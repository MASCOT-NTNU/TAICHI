"""
This script interpolates data from sinmod onto the coordinates
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
#% Step I: load coordinates to extract data from SINMOD
from DataHandler.SINMOD import SINMOD
from MAFIA.Simulation.Config.Config import *

lats = np.load(FILEPATH + "models/lats_small.npy").reshape(-1, 1)
lons = np.load(FILEPATH + "models/lons_small.npy").reshape(-1, 1)
depth = np.load(FILEPATH + "models/depth_small.npy").reshape(-1, 1)
# lats = np.load(FILEPATH + "models/lats.npy").reshape(-1, 1)
# lons = np.load(FILEPATH + "models/lons.npy").reshape(-1, 1)
# depth = np.load(FILEPATH + "models/depth.npy").reshape(-1, 1)
coordinates = np.hstack((lats, lons, depth))

# == get desired sinmod data
SINMODPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
# files = os.listdir(SINMODPATH)
# files.sort()
# filenames = []
# for file in files:
#     if file.endswith('.nc'):
#         # print(file)
#         filenames.append(file)
# filenames = filenames[-4:]
# filenames_fullpath = []
# for file in filenames:
#     filenames_fullpath.append(SINMODPATH+file)
filenames_fullpath = None
# ==

sinmod = SINMOD()
sinmod.load_sinmod_data(raw_data=True, filenames=filenames_fullpath)
# sinmod.get_data_at_coordinates(coordinates)

#%%
DATAPATH = FILEPATH+"Simulation/Config/Data/"
#% Step II: extract data by section
p1 = coordinates[0:5000,:]
sinmod.get_data_at_coordinates(p1, filename=DATAPATH+'p1.csv')

#%%
p2 = coordinates[5000:10000,:]
sinmod.get_data_at_coordinates(p2, filename=DATAPATH+'p2.csv')
os.system('say complete 2')
#%%
p3 = coordinates[10000:-1,:]
sinmod.get_data_at_coordinates(p3, filename=DATAPATH+'p3.csv')
os.system('say complete 3')
# p4 = coordinates[15000:20000,:]
# sinmod.get_data_at_coordinates(p4, filename=DATAPATH+'p4.csv')
# os.system('say complete 4')
# p5 = coordinates[20000:,:]
# sinmod.get_data_at_coordinates(p5, filename=DATAPATH+'p5.csv')
# os.system('say complete 5')
#%%
#% Step III: save data to scv section by section
datapath = FILEPATH+"Simulation/Config/Data/"
import os
import pandas as pd

files = os.listdir(datapath)
# for file in files:
file = files[0]
df1 = pd.read_csv(datapath+file)

file = files[1]
df2 = pd.read_csv(datapath+file)

file = files[2]
df3 = pd.read_csv(datapath+file)
#%%
# file = files[3]
# df4 = pd.read_csv(datapath+file)
#
# file = files[4]
# df5 = pd.read_csv(datapath+file)

# df = np.vstack((df1, df2, df3, df4, df5))
df = pd.concat([df1, df2, df3], ignore_index=True, sort=False)
df.to_csv(datapath + "data_mu_truth.csv", index=False)
os.system('say complete all')
#%%

# import matplotlib.pyplot as plt
# plt.scatter(df[:, 1], df[:, 0], c=df[:, 2], cmap='RdBu', vmin=16, vmax=32)
# plt.colorbar()
# plt.show()
import plotly.graph_objects as go
import numpy as np

# Helix equation
t = np.linspace(0, 20, 100)
x = df[:, 1]
y = df[:, 0]
z = df[:, 2]
value = df[:, 3]

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=-z,
    mode='markers',
    marker=dict(
        size=12,
        color=value,                # set color to an array/list of desired values
        colorscale='BrBG',   # choose a colorscale
        # opacity=0.8,
        cmin=16,
        cmax=32
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
import plotly
plotly.offline.plot(fig, filename=FILEPATH+"/fig/data.html", auto_open=True)

