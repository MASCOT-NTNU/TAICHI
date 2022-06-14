"""
This script simulates Agent Yin's behaviour
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-14
"""
import numpy as np

from usr_func import *
from TAICHI.Simulation.Config.Config import *
from TAICHI.Simulation.PlanningStrategies.Myopic3D import MyopicPlanning3D
from TAICHI.Simulation.Knowledge.Knowledge import Knowledge
from TAICHI.spde import spde
import pickle
import concurrent.futures


class Agent:

    def __init__(self, name="Agent"):
        self.agent_name = name
        self.load_waypoint()
        self.load_gmrf_grid()
        self.load_gmrf_model()
        self.load_prior()
        self.load_simulated_truth()
        self.update_knowledge()
        self.load_hash_neighbours()
        self.load_hash_waypoint2gmrf()
        self.initialise_function_calls()
        self.setup_data_container()
        print("S1-S10 complete!" + self.agent_name + " is initialised successfully!")

    def load_waypoint(self):
        self.waypoints = pd.read_csv(FILEPATH + "Simulation/Config/WaypointGraph.csv").to_numpy()
        print("S1: Waypoint is loaded successfully!")

    def load_gmrf_grid(self):
        self.gmrf_grid = pd.read_csv(FILEPATH + "Simulation/Config/GMRFGrid.csv").to_numpy()
        self.N_gmrf_grid = len(self.gmrf_grid)
        print("S2: GMRF grid is loaded successfully!")

    def load_gmrf_model(self):
        self.gmrf_model = spde(model=2, reduce=True, method=2)
        print("S3: GMRF model is loaded successfully!")

    def load_prior(self):
        print("S4: Prior is loaded successfully!")
        pass

    def load_simulated_truth(self):
        path_mu_truth = FILEPATH + "Simulation/Config/Data/data_mu_truth.csv"
        self.simulated_truth = pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1)
        print("S5: Simulated truth is loaded successfully!")

    def update_knowledge(self):
        self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
        print("S6: Knowledge of the field is set up successfully!")

    def load_hash_neighbours(self):
        neighbour_file = open(FILEPATH + "Simulation/Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(neighbour_file)
        neighbour_file.close()
        print("S7: Neighbour hash table is loaded successfully!")

    def load_hash_waypoint2gmrf(self):
        waypoint2gmrf_file = open(FILEPATH + "Simulation/Config/HashWaypoint2GMRF.p", 'rb')
        self.hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
        waypoint2gmrf_file.close()
        print("S8: Waypoint2GMRF hash table is loaded successfully!")

    def initialise_function_calls(self):
        get_ind_at_location3d_xyz(self.waypoints, 1, 2, 3)  # used to initialise the function call
        print("S9: Function calls are initialised successfully!")

    def setup_data_container(self):
        self.data_agent = np.empty([0, 2])
        print("S10: Data container is initialised successfully!")

    def set_starting_location(self, starting_location):
        lat, lon, depth = starting_location
        x, y = latlon2xy(lat, lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.x_start = x
        self.y_start = y
        self.z_start = depth
        print("Starting location is set up successfully!")

    def prepare_run(self):
        self.ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, self.x_start, self.y_start, self.z_start)
        self.ind_previous_waypoint = self.ind_current_waypoint
        self.ind_pioneer_waypoint = self.ind_current_waypoint
        self.ind_next_waypoint = self.ind_current_waypoint
        self.ind_visited_waypoint = []
        self.ind_visited_waypoint.append(self.ind_current_waypoint)

        self.myopic3d_planner = MyopicPlanning3D(waypoints=self.waypoints, hash_neighbours=self.hash_neighbours,
                                                 hash_waypoint2gmrf=self.hash_waypoint2gmrf)
        folder = FILEPATH + "Simulation/Waypoint/" + self.agent_name
        checkfolder(folder)
        self.filename_ind_next = folder + "/ind_next.txt"

        self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                            self.ind_current_waypoint,
                            self.ind_previous_waypoint,
                            self.ind_visited_waypoint,
                            filename=self.filename_ind_next)
        self.ind_next_waypoint = int(np.loadtxt(self.filename_ind_next))
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                            self.ind_next_waypoint,
                            self.ind_current_waypoint,
                            self.ind_visited_waypoint,
                            filename=self.filename_ind_next)
        self.ind_pioneer_waypoint = int(np.loadtxt(self.filename_ind_next))
        print("Ready to run!")

    def run(self, step=0, share=False, agent_location=None):
        if not share:
            self.ind_sample_gmrf = self.get_ind_sample(self.ind_previous_waypoint, self.ind_current_waypoint)
            self.salinity_measured = self.simulated_truth[self.ind_sample_gmrf]
            print("Salinity: ", self.salinity_measured.shape)
            print("ind sample: ", self.ind_sample_gmrf.shape)

            self.data_agent = np.append(self.data_agent, np.hstack((vectorise(self.ind_sample_gmrf),
                                                                    vectorise(self.salinity_measured))),
                                         axis=0)
        else:
            x_agent, y_agent = agent_location
            self.ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, x_agent, y_agent, .5) # surface
            self.ind_previous_waypoint = self.ind_current_waypoint
            self.ind_sample_gmrf = self.get_ind_sample(self.ind_previous_waypoint, self.ind_current_waypoint)
            self.salinity_measured = self.simulated_truth[self.ind_sample_gmrf]

            self.ind_measured_by_other_agent = self.data_from_other_agent[:, 0]
            self.salinity_measured_from_other_agent = vectorise(self.data_from_other_agent[:, 1])

            self.ind_sample_gmrf = np.append(self.ind_measured_by_other_agent, self.ind_sample_gmrf, axis=0)
            self.salinity_measured = np.append(self.salinity_measured_from_other_agent, self.salinity_measured, axis=0)

        t1 = time.time()
        self.gmrf_model.update(rel=self.salinity_measured, ks=self.ind_sample_gmrf)
        t2 = time.time()
        print("Update consumed: ", t2 - t1)

        self.knowledge.mu = self.gmrf_model.mu
        self.knowledge.SigmaDiag = self.gmrf_model.mvar()

        self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
                            self.ind_next_waypoint,
                            self.ind_current_waypoint,
                            self.ind_visited_waypoint,
                            filename=self.filename_ind_next)
        self.ind_pioneer_waypoint = int(np.loadtxt(self.filename_ind_next))

        # == plot gmrf section
        xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
        yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
        zrot = -self.gmrf_grid[:, 2]

        ind_plot = np.where((zrot<0) * (zrot>=-5) * (xrot>70))[0]
        mu_plot = self.knowledge.mu[ind_plot]
        var_plot = self.knowledge.SigmaDiag[ind_plot]
        ep_plot = get_excursion_prob(mu_plot.astype(np.float32), var_plot.astype(np.float32),
                                     np.float32(self.gmrf_model.threshold))
        self.yplot = xrot[ind_plot]
        self.xplot = yrot[ind_plot]
        self.zplot = zrot[ind_plot]

        # fig = go.Figure(data=go.Scatter3d(
        #     x=self.xplot,
        #     y=self.yplot,
        #     z=self.zplot,
        #     mode='markers',
        #     marker=dict(color=mu_plot, size=2, opacity=.0)
        # ))
        # fig.add_trace(go.Scatter3d(
        #     x=self.xplot,
        #     y=self.yplot,
        #     z=self.zplot,
        #     mode='markers',
        #     marker=dict(color=mu_plot)
        # ))
        # fig.add_trace(go.Scatter3d(
        #     x=self.waypoints[:, 1],
        #     y=self.waypoints[:, 0],
        #     z=-self.waypoints[:, 2],
        #     mode='markers',
        #     marker=dict(color='black', size=1, opacity=.1)
        # ))
        figpath = FIGPATH + self.agent_name + "/birdview/"
        filename = figpath + "mean/P_{:03d}.jpg".format(step)
        fig_mu = self.plot_figure(mu_plot, filename, vmin=10, vmax=30, opacity=.4,
                                  surface_count=6, cmap="BrBG", cbar_title="Salinity", share=share)

        # filename = figpath + "var/P_{:03d}.jpg".format(step)
        # fig_var = self.plot_figure(var_plot, filename, vmin=0, vmax=1, opacity=.4,
        #                           surface_count=4, cmap="RdBu", cbar_title="STD", share=share)
        #
        # filename = figpath + "ep/P_{:03d}.jpg".format(step)
        # fig_ep = self.plot_figure(ep_plot, filename, vmin=0, vmax=1, opacity=.4,
        #                           surface_count=10, cmap="Brwnyl", cbar_title="EP", share=share)

        self.ind_previous_waypoint = self.ind_current_waypoint
        self.ind_current_waypoint = self.ind_next_waypoint
        self.ind_next_waypoint = self.ind_pioneer_waypoint
        self.ind_visited_waypoint.append(self.ind_current_waypoint)
        print("previous ind: ", self.ind_previous_waypoint)
        print("current ind: ", self.ind_current_waypoint)
        print("next ind: ", self.ind_next_waypoint)
        print("pioneer ind: ", self.ind_pioneer_waypoint)

        if step == NUM_STEPS-1:
            plotly.offline.plot(fig_mu, filename=figpath + "mean/P_mean.html", auto_open=True)
            # plotly.offline.plot(fig_var, filename=figpath + "var/P_var.html", auto_open=True)
            # plotly.offline.plot(fig_ep, filename=figpath + "ep/P_ep.html", auto_open=True)

    def save_agent_data(self):
        datapath = FILEPATH + "Simulation/AgentsData/" + self.agent_name + ".npy"
        np.save(datapath, self.data_agent)
        print("Data from " + self.agent_name + " is saved successfully!")

    def load_data_from_agent(self, agent_name="Agent"):
        datapath = FILEPATH + "Simulation/AgentsData/" + agent_name + ".npy"
        if not os.path.exists(datapath):
            self.data_from_other_agent = np.empty(0, 2)
        else:
            self.data_from_other_agent = np.load(datapath)
        print("Data from " + agent_name + " is loaded successfully!")

    def clear_agent_data(self):
        self.data_agent = np.empty([0, 2])
        print("Data from " + self.agent_name + " is cleared successfully!", self.data_agent)

    def get_ind_sample(self, ind_start, ind_end):
        N = 20
        x_start, y_start, z_start = self.waypoints[ind_start, :]
        x_end, y_end, z_end = self.waypoints[ind_end, :]
        x_path = np.linspace(x_start, x_end, N)
        y_path = np.linspace(y_start, y_end, N)
        z_path = np.linspace(z_start, z_end, N)
        dataset = np.vstack((x_path, y_path, z_path, np.zeros_like(z_path))).T
        ind, value = self.assimilate_data(dataset)
        return ind

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[:10, :])
        ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= .25)[0]
        dataset = dataset[ind_remove_noise_layer, :]
        print("dataset after filtering: ", dataset[:10, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 1]).T) ** 2
        dz = ((vectorise(dataset[:, 2]) @ np.ones([1, self.N_gmrf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 2]).T) * GMRF_DISTANCE_NEIGHBOUR) ** 2
        dist = dx + dy + dz
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return ind_assimilated, vectorise(salinity_assimilated)

    def plot_figure(self, value, filename, vmin=None, vmax=None, opacity=None, surface_count=None, cmap=None,
                    cbar_title=None, share=False):
        points_int, values_int = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
        fig = go.Figure(data=go.Volume(
            x=points_int[:, 0],
            y=points_int[:, 1],
            z=points_int[:, 2],
            value=values_int.flatten(),
            isomin=vmin,
            isomax=vmax,
            opacity=opacity,
            surface_count=surface_count,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
        )

        # == plot waypoint section
        xrot = self.waypoints[:, 0] * np.cos(ROTATED_ANGLE) - self.waypoints[:, 1] * np.sin(ROTATED_ANGLE)
        yrot = self.waypoints[:, 0] * np.sin(ROTATED_ANGLE) + self.waypoints[:, 1] * np.cos(ROTATED_ANGLE)
        zrot = -self.waypoints[:, 2]
        fig.add_trace(go.Scatter3d(name="Waypoint graph",
            x=yrot,
            y=xrot,
            z=zrot,
            mode='markers',
            marker=dict(color='black', size=1, opacity=.1)
        ))
        fig.add_trace(go.Scatter3d(name="Previous waypoint",
            x=[yrot[self.ind_previous_waypoint]],
            y=[xrot[self.ind_previous_waypoint]],
            z=[zrot[self.ind_previous_waypoint]],
            mode='markers',
            marker=dict(color='yellow', size=10)
        ))
        fig.add_trace(go.Scatter3d(name="Current waypoint",
            x=[yrot[self.ind_current_waypoint]],
            y=[xrot[self.ind_current_waypoint]],
            z=[zrot[self.ind_current_waypoint]],
            mode='markers',
            marker=dict(color='red', size=10)
        ))
        fig.add_trace(go.Scatter3d(name="Next waypoint",
            x=[yrot[self.ind_next_waypoint]],
            y=[xrot[self.ind_next_waypoint]],
            z=[zrot[self.ind_next_waypoint]],
            mode='markers',
            marker=dict(color='blue', size=10)
        ))
        fig.add_trace(go.Scatter3d(name="Pioneer waypoint",
            x=[yrot[self.ind_pioneer_waypoint]],
            y=[xrot[self.ind_pioneer_waypoint]],
            z=[zrot[self.ind_pioneer_waypoint]],
            mode='markers',
            marker=dict(color='green', size=10)
        ))
        fig.add_trace(go.Scatter3d(name="Visited waypoints",
            x=yrot[self.ind_visited_waypoint],
            y=xrot[self.ind_visited_waypoint],
            z=zrot[self.ind_visited_waypoint],
            mode='markers+lines',
            marker=dict(color='black', size=4),
            line=dict(color='black', width=3)
        ))
        if share:
            ind_measured_by_other_agent = self.ind_measured_by_other_agent.astype(int)
            fig.add_trace(go.Scatter3d(name="Shared waypoints",
                                       x=self.gmrf_grid[ind_measured_by_other_agent, 1],
                                       y=self.gmrf_grid[ind_measured_by_other_agent, 0],
                                       z=-self.gmrf_grid[ind_measured_by_other_agent, 2],
                                       mode='markers',
                                       marker=dict(color='cyan', size=4,
                                                   opacity=.5),
                                       ))
        # fig.add_trace(go.Scatter3d(
        #     x=yrot[self.myopic3d_planner.ind_candidates],
        #     y=xrot[self.myopic3d_planner.ind_candidates],
        #     z=zrot[self.myopic3d_planner.ind_candidates],
        #     mode='markers',
        #     marker=dict(color='orange', size=5, opacity=.3)
        # ))
        fig.update_coloraxes(colorscale=cmap, colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                              tickfont=dict(size=18, family="Times New Roman"),
                                                              title="Salinity",
                                                              titlefont=dict(size=18, family="Times New Roman")),
                             colorbar_title=cbar_title)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2)
        )
        fig.update_layout(coloraxis_colorbar_x=0.8)
        fig.update_layout(
            title={
                'text': "Adaptive 3D myopic illustration",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=30, family="Times New Roman"),
            },
            scene=dict(
                zaxis=dict(nticks=4, range=[-5.5, -.5], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
            scene_camera=camera,
        )

        # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)
        wd = os.path.dirname(filename)
        checkfolder(wd)
        # if not os.path.exists(wd):
        #     pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
        #     print(wd + " is created successfully!")
        fig.write_image(filename, width=1980, height=1080)
        return fig

    def check_agent(self):
        self.set_starting_location(AGENT1_START_LOCATION)
        self.prepare_run()
        self.run()
        pass


if __name__ == "__main__":
    a = Agent()
    a.check_agent()





#%%

import plotly.graph_objects as go
import numpy as np

# Helix equation
t = np.linspace(0, 20, 100)
x, y, z = np.cos(t), np.sin(t), t

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers + lines',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        # opacity=0.8
    ),
    opacity=.4
    # line=dict(
    # width=2,
    # ),

)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename=FIGPATH + "opacity.html", auto_open=True)

