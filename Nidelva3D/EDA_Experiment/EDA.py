"""
EDA for the experiment data analysis

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-07-05
"""

from WGS import WGS
from Planner.Myopic3D import Myopic3D
from usr_func.checkfolder import checkfolder
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from pykdtree.kdtree import KDTree
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec


class EDA:

    def __init__(self) -> None:
        # Set up file path
        self.filepath = os.getcwd() + "/csv/AUVData.csv"
        self.figpath = os.getcwd() + "/../fig/Experiment/"
        checkfolder(self.figpath)

        # Set up GMRF kernel planner
        self.myopic_gmrf = Myopic3D(kernel="GMRF")
        self.waypoints = self.myopic_gmrf.wp.get_waypoints()
        self.waypoints_tree = KDTree(self.waypoints)
        self.depths = np.unique(self.waypoints[:, 2])

        # Set up GRF kernel planner
        self.myopic_grf = Myopic3D(kernel="GRF")

        # Load data
        self.load_data()

    def load_data(self) -> None:
        self.data_auv_wgs = pd.read_csv(self.filepath)
        x, y = WGS.latlon2xy(self.data_auv_wgs['lat'], self.data_auv_wgs['lon'])
        self.data_auv = np.stack((x, y, self.data_auv_wgs['depth'], self.data_auv_wgs['salinity']), axis=1)
        self.visited_locs = self.data_auv[:, :-1]

    def rerun_mission_using_grf_kernel(self) -> None:
        dist = []
        counter = 0
        prev_loc = self.waypoints[0, :]
        ind_prev = 0
        ind_grf_prev = 0
        for i in range(self.visited_locs.shape[0]):
            dist_temp, ind = self.waypoints_tree.query(self.visited_locs[i, :].reshape(1, -1))
            dist.append(dist_temp)
            ind = ind[0]

            if (dist_temp < 5) and (self.cal_distance_between_two_locs(prev_loc, self.visited_locs[i, :]) > 10):
                ctd_data = self.data_auv[ind_prev:i, :]

                print("CTD: ", ctd_data)

                # Section to use GRF to plan the next waypoint
                self.myopic_grf.set_current_index(ind_grf_prev)
                self.myopic_grf.set_next_index(ind)

                self.myopic_grf.kernel.assimilate_data(ctd_data)
                self.myopic_grf.update_planner()
                self.myopic_grf.get_pioneer_waypoint_index()

                ind_grf = self.myopic_grf.get_next_index()
                print("ind_grf_prev: ", ind_grf_prev)
                print("ind_grf: ", ind_grf)
                print("ind_grf_next: ", self.myopic_grf.get_next_index())

                self.plot_neighbour_on_waypoint_graph_2d(self.visited_locs[:i, :], self.waypoints[ind, :],
                                                         self.waypoints[ind_grf, :], counter)

                counter += 1
                ind_prev = i
                prev_loc = self.visited_locs[i, :]

        print(counter)

    @staticmethod
    def cal_distance_between_two_locs(loc1, loc2) -> float:
        dx = loc1[0] - loc2[0]
        dy = loc1[1] - loc2[1]
        dz = loc1[2] - loc2[2]
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def plot_neighbour_on_waypoint_graph_2d(self, visited_locs, activated_waypoints,grf_waypoints, counter) -> None:
        print("counter: ", counter)

        plt.figure(figsize=(54, 8))
        gs = GridSpec(1, 6, figure=plt.gcf())

        def plot_subplot(i):
            ax = plt.subplot(gs[i])
            ind_layer = np.where(self.waypoints[:, 2] == self.depths[i])[0]
            ax.plot(self.waypoints[ind_layer, 1], self.waypoints[ind_layer, 0], 'k.', alpha=.3)
            ind_depth = np.where(np.abs(visited_locs[:, 2] - self.depths[i]) < .25)[0]
            ax.plot(visited_locs[:, 1], visited_locs[:, 0], 'b-', alpha=.5)
            ax.plot(visited_locs[ind_depth, 1], visited_locs[ind_depth, 0], 'b.')
            if activated_waypoints[2] == self.depths[i]:
                ax.plot(activated_waypoints[1], activated_waypoints[0], 'r.', markersize=10)
            if grf_waypoints[2] == self.depths[i]:
                ax.plot(grf_waypoints[1], grf_waypoints[0], 'g.', markersize=10)
            ax.axis('equal')
            ax.set_title("Depth: {:.2f}".format(self.depths[i]))

        for i in range(6):
            plot_subplot(i)

        plt.savefig(self.figpath + "P_{:03d}.png".format(counter))
        plt.close("all")

    def plot_neighbour_on_waypoint_graph(self, visited_locs, activated_waypoints, counter) -> None:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])

        fig.add_trace(
            go.Scatter3d(
                x=self.waypoints[:, 1],
                y=self.waypoints[:, 0],
                z=-self.waypoints[:, 2],
                mode="markers",
                marker=dict(
                    size=10,
                    color="black",
                    opacity=0.5,
                ),
                name="Waypoints",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=visited_locs[:, 1],
                y=visited_locs[:, 0],
                z=-visited_locs[:, 2],
                mode="markers + lines",
                marker=dict(
                    size=1,
                    color="blue",
                    opacity=0.5,
                ),
                line=dict(
                    color="yellow",
                    width=2,
                ),
                name="Visited Locations",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter3d(
                x=activated_waypoints[:, 1],
                y=activated_waypoints[:, 0],
                z=-activated_waypoints[:, 2],
                mode="markers",
                marker=dict(
                    size=20,
                    color="red",
                ),
                name="Activated Waypoints",
            ),
            row=1,
            col=1,
        )

        # fig.update_layout(
        #     title="Waypoints",
        #     autosize=False,
        #     width=1000,
        #     height=1000,
        #     margin=dict(l=65, r=50, b=65, t=90),
        #     scene=dict(
        #         xaxis=dict(title="East"),
        #         yaxis=dict(title="North"),
        #         zaxis=dict(title="Depth"),
        #     ),
        #     legend=dict(
        #         yanchor="top",
        #         y=0.99,
        #         xanchor="left",
        #         x=0.01,
        #     ),
        # )

        plotly.offline.plot(fig, filename=self.figpath + "P_{:03d}.html".format(counter), auto_open=False)


if __name__ == "__main__":
    e = EDA()
