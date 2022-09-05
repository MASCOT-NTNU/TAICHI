"""
Visualiser object handles the planning visualisation part.
"""

from Agent import Agent
import os
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from usr_func.interpolate_3d import interpolate_3d
from usr_func.vectorize import vectorize


class Visualiser:

    agent = None
    figpath = os.getcwd() + "../fig/Myopic3D/"

    def __init__(self):
        pass

    def update_agent(self, agent: Agent) -> None:
        self.agent = agent
        self.cnt = self.agent.get_counter()
        self.myopic = self.agent.myopic
        self.gmrf = self.myopic.gmrf
        self.grid = self.myopic.gmrf.get_gmrf_grid()

        self.plot_agent()

    def plot_agent(self):
        vmin = 0
        vmax = 28
        filename = self.figpath + "test"
        # html = html
    
        ind_remove_top_layer = np.where(self.grid[:, 2] > 0)[0]
        xgrid = self.grid[ind_remove_top_layer, 0]
        ygrid = self.grid[ind_remove_top_layer, 1]
        rotated_angle = self.gmrf.get_rotated_angle()
        xrotated = xgrid * np.cos(rotated_angle) - ygrid * np.sin(rotated_angle)
        yrotated = xgrid * np.sin(rotated_angle) + ygrid * np.cos(rotated_angle)

        RR = np.array([[np.cos(rotated_angle), -np.sin(rotated_angle), 0],
                       [np.sin(rotated_angle), np.cos(rotated_angle), 0],
                       [0, 0, 1]])

        xplot = yrotated
        yplot = xrotated

        mu = self.gmrf.get_mu()
        mvar = self.gmrf.get_mvar()
        mu[mu < 0] = 0
        ind_selected_to_plot = np.where(mu[ind_remove_top_layer] >= 0)[0]
        xplot = xplot[ind_selected_to_plot]
        yplot = yplot[ind_selected_to_plot]
        zplot = -self.grid[ind_remove_top_layer, 2][ind_selected_to_plot]

        points_mean, values_mean = interpolate_3d(xplot, yplot, zplot, mu[ind_remove_top_layer][ind_selected_to_plot])

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        fig.add_trace(go.Volume(
            x=points_mean[:, 0],
            y=points_mean[:, 1],
            z=points_mean[:, 2],
            value=values_mean,
            isomin=vmin,
            isomax=vmax,
            opacity=.3,
            surface_count=10,
            colorscale="BrBG",
            # coloraxis="coloraxis1",
            # colorbar=dict(x=0.5, y=0.5, len=.5),
            # reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))

        # fig.add_trace(go.Scatter3d(
        #     x=xplot,
        #     y=yplot,
        #     z=zplot,
        #     mode='markers',
        #     marker=dict(
        #         size=12,
        #         color=self.knowledge.mu_cond[ind_remove_top_layer][ind_selected_to_plot],  # set color to an array/list of desired values
        #         colorscale='BrBG',  # choose a colorscale
        #         showscale=True,
        #         opacity=0.1
        #     )
        # ))

        # if len(self.knowledge.ind_neighbour_filtered_waypoint):
        #     fig.add_trace(go.Scatter3d(
        #         x=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 1],
        #         y=self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 0],
        #         z=-self.knowledge.coordinates_waypoint[self.knowledge.ind_neighbour_filtered_waypoint, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=15,
        #             color="white",
        #             showscale=False,
        #         ),
        #         showlegend=False,
        #     ),
        #         row='all', col='all'
        #     )

        id = self.myopic.get_current_index()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        wp = RR @ vectorize(wp)
        fig.add_trace(go.Scatter3d(
            x=[wp[1]],
            y=[wp[0]],
            z=[-wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="red",
                showscale=False,
            ),
            showlegend=False,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_next_index()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        wp = RR @ vectorize(wp)
        fig.add_trace(go.Scatter3d(
            x=[wp[1]],
            y=[wp[0]],
            z=[-wp[2]],
            mode='markers',
            marker=dict(
                size=20,
                color="blue",
                showscale=False,
            ),
            showlegend=False,  # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        id = self.myopic.get_trajectory_indices()
        wp = self.myopic.wp.get_waypoint_from_ind(id)
        wp = wp @ RR
        fig.add_trace(go.Scatter3d(
            # print(trajectory),
            x=wp[:, 0],
            y=wp[:, 1],
            z=-wp[:, 2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color="black",
                showscale=False,
            ),
            line=dict(
                color="yellow",
                width=3,
                showscale=False,
            ),
            showlegend=False,
        ),
            row='all', col='all'
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            title={
                'text': "Simulation",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene=dict(
                zaxis=dict(nticks=4, range=[-10, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
        )

        plotly.offline.plot(fig, filename=filename + ".html", auto_open=False)
