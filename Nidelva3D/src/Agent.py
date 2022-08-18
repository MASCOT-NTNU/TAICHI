# """
# This script simulates Agent Yin's behaviour
# Author: Yaolin Ge
# Contact: yaolin.ge@ntnu.no
# Date: 2022-06-14
# """
#
# from usr_func import *
# from TAICHI.Nidelva3D.Config.Config import *
# from TAICHI.Nidelva3D.PlanningStrategies.Myopic3D import MyopicPlanning3D
# from TAICHI.Nidelva3D.Knowledge.Knowledge import Knowledge
# from TAICHI.Nidelva3D.spde import spde
# import pickle
# import concurrent.futures
# from sklearn.metrics import mean_squared_error
# import properscoring
#
#
# class Agent:
#
#     def __init__(self, name="Agent", plot=False):
#         self.agent_name = name
#         self.plot = plot
#         self.load_waypoint()
#         self.load_gmrf_grid()
#         self.load_gmrf_model()
#         self.load_prior()
#         self.load_simulated_truth()
#         self.update_knowledge()
#         self.load_hash_neighbours()
#         self.load_hash_waypoint2gmrf()
#         self.initialise_function_calls()
#         self.setup_data_container()
#         print("S1-S10 complete!" + self.agent_name + " is initialised successfully!")
#
#     def load_waypoint(self):
#         self.waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
#         print("S1: Waypoint is loaded successfully!")
#
#     def load_gmrf_grid(self):
#         self.gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
#         self.N_gmrf_grid = len(self.gmrf_grid)
#         print("S2: GMRF grid is loaded successfully!")
#
#     def load_gmrf_model(self):
#         self.gmrf_model = spde(model=2, method=2)
#         print("S3: GMRF model is loaded successfully!")
#
#     def load_prior(self):
#         print("S4: Prior is loaded successfully!")
#         pass
#
#     def load_simulated_truth(self):
#         path_mu_truth = FILEPATH + "Config/Data/data_mu_truth.csv"
#         weight = 1
#         self.simulated_truth = (weight * pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1) +
#                                 np.random.rand(len(self.gmrf_model.mu)).reshape(-1, 1))
#                                 # (1 - weight) * vectorise(self.gmrf_model.mu))  #TODO:
#         # self.simulated_truth = pd.read_csv(path_mu_truth).to_numpy()[:, -1].reshape(-1, 1)
#         print("S5: Simulated truth is loaded successfully!")
#
#     def update_knowledge(self):
#         self.knowledge = Knowledge(gmrf_grid=self.gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
#         print("S6: Knowledge of the field is set up successfully!")
#
#     def load_hash_neighbours(self):
#         neighbour_file = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
#         self.hash_neighbours = pickle.load(neighbour_file)
#         neighbour_file.close()
#         print("S7: Neighbour hash table is loaded successfully!")
#
#     def load_hash_waypoint2gmrf(self):
#         waypoint2gmrf_file = open(FILEPATH + "Config/HashWaypoint2GMRF.p", 'rb')
#         self.hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
#         waypoint2gmrf_file.close()
#         print("S8: Waypoint2GMRF hash table is loaded successfully!")
#
#     def initialise_function_calls(self):
#         get_ind_at_location3d_xyz(self.waypoints, 1, 2, 3)  # used to initialise the function call
#         print("S9: Function calls are initialised successfully!")
#
#     def setup_data_container(self):
#         self.data_agent = np.empty([0, 2])
#         self.ibv = []
#         self.uncertainty = []
#         self.crps = []
#         self.rmse = []
#         print("S10: Data container is initialised successfully!")
#
#     def prepare_run(self, starting_location=None, ind_legal=None):
#         lat, lon, depth = starting_location
#         x, y = latlon2xy(lat, lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
#         print("Starting location is set up successfully!")
#         self.ind_current_waypoint = get_ind_at_location3d_xyz(self.waypoints, x, y, depth)
#         print("starting index: ", self.ind_current_waypoint)
#
#         self.ind_previous_waypoint = self.ind_current_waypoint
#         self.ind_pioneer_waypoint = self.ind_current_waypoint
#         self.ind_next_waypoint = self.ind_current_waypoint
#         self.ind_visited_waypoint = []
#         self.ind_visited_waypoint.append(self.ind_current_waypoint)
#
#         self.myopic3d_planner = MyopicPlanning3D(waypoints=self.waypoints, hash_neighbours=self.hash_neighbours,
#                                                  hash_waypoint2gmrf=self.hash_waypoint2gmrf)
#         folder = FILEPATH + "Waypoint/" + self.agent_name
#         checkfolder(folder)
#         self.filename_ind_next = folder + "/ind_next.txt"
#
#         if ind_legal is None:
#             ind_legal = np.arange(self.waypoints.shape[0])
#
#         self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model, ind_legal=ind_legal)
#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
#                             self.ind_current_waypoint,
#                             self.ind_previous_waypoint,
#                             self.ind_visited_waypoint,
#                             filename=self.filename_ind_next)
#         self.ind_next_waypoint = int(np.loadtxt(self.filename_ind_next))
#         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#             executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
#                             self.ind_next_waypoint,
#                             self.ind_current_waypoint,
#                             self.ind_visited_waypoint,
#                             filename=self.filename_ind_next)
#         self.ind_pioneer_waypoint = int(np.loadtxt(self.filename_ind_next))
#         print("Ready to run!")
#
#         # save prior and ground truth
#         xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
#         yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
#         zrot = -self.gmrf_grid[:, 2]
#
#         ind_plot = np.where((zrot<0) * (zrot>=-5) * (xrot>70))[0]
#         mu_plot = self.knowledge.mu[ind_plot]
#         var_plot = self.knowledge.SigmaDiag[ind_plot]
#         ep_plot = get_excursion_prob(mu_plot.astype(np.float32), var_plot.astype(np.float32),
#                                      np.float32(self.gmrf_model.threshold))
#         self.yplot = xrot[ind_plot]
#         self.xplot = yrot[ind_plot]
#         self.zplot = zrot[ind_plot]
#
#         if self.plot:
#             figpath = FIGPATH + self.agent_name + "/"
#             filename = figpath + "mean/Prior.jpg"
#             fig_mu = self.plot_figure(mu_plot, filename, vmin=10, vmax=27, opacity=.4,
#                                       surface_count=6, cmap="BrBG", cbar_title="Salinity")
#             filename = figpath + "mean/Prior.html"
#             wd = os.path.dirname(filename)
#             checkfolder(wd)
#             plotly.offline.plot(fig_mu, filename=filename, auto_open=True)
#
#             figpath = FIGPATH + self.agent_name + "/"
#             filename = figpath + "mean/GroundTruth.jpg"
#             mu_plot = self.simulated_truth[ind_plot]
#             fig_mu = self.plot_figure(mu_plot, filename, vmin=10, vmax=27, opacity=.4,
#                                       surface_count=6, cmap="BrBG", cbar_title="Salinity")
#             filename = figpath + "mean/GroundTruth.html"
#             wd = os.path.dirname(filename)
#             checkfolder(wd)
#             plotly.offline.plot(fig_mu, filename=filename, auto_open=True)
#
#     def update_pioneer_waypoint(self, waypoint_location=None):
#         x, y = waypoint_location
#         self.ind_pioneer_waypoint = get_ind_at_location3d_xyz(self.waypoints, x, y, .5)  # surface
#         print("ind pioneer waypoint is updated successfully!", self.ind_pioneer_waypoint)
#
#     def sample(self):
#         self.ind_sample_gmrf = self.get_ind_sample(self.ind_previous_waypoint, self.ind_current_waypoint)
#         self.salinity_measured = self.simulated_truth[self.ind_sample_gmrf]
#         print("Salinity: ", self.salinity_measured.shape)
#         print("ind sample: ", self.ind_sample_gmrf.shape)
#         self.data_agent = np.append(self.data_agent, np.hstack((vectorise(self.ind_sample_gmrf),
#                                                                 vectorise(self.salinity_measured))),
#                                     axis=0)
#
#     def monitor_data(self):
#         truth = self.simulated_truth
#         truth[truth < 0] = 0
#         self.rmse.append(mean_squared_error(self.simulated_truth, self.knowledge.mu, squared=True))
#         self.uncertainty.append(np.sum(self.knowledge.SigmaDiag))
#         self.ibv.append(self.myopic3d_planner.get_eibv_from_gmrf_model(np.array([self.ind_current_waypoint])))
#         self.crps.append(np.sum(properscoring.crps_gaussian(self.simulated_truth[self.ind_current_waypoint],
#                                                             self.knowledge.mu, self.knowledge.SigmaDiag)))
#
#     def run(self, step=0, pre_share=False, share=False, other_agent=None, ind_legal=None):
#         self.monitor_data()
#         if share:
#             self.ind_measured_by_other_agent = self.data_from_other_agent[:, 0]
#             self.salinity_measured_from_other_agent = vectorise(self.data_from_other_agent[:, 1])
#
#             self.ind_sample_gmrf = np.append(self.ind_measured_by_other_agent, self.ind_sample_gmrf, axis=0)
#             self.salinity_measured = np.append(self.salinity_measured_from_other_agent, self.salinity_measured, axis=0)
#
#         t1 = time.time()
#         self.gmrf_model.update(rel=self.salinity_measured, ks=self.ind_sample_gmrf)
#         t2 = time.time()
#         print("Update consumed: ", t2 - t1)
#
#         self.knowledge.mu = self.gmrf_model.mu
#         self.knowledge.SigmaDiag = self.gmrf_model.mvar()
#
#         if not pre_share:
#             if ind_legal is None:
#                 ind_legal = np.arange(self.waypoints.shape[0])
#             self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model,
#                                                  ind_legal=ind_legal)
#             with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#                 executor.submit(self.myopic3d_planner.find_next_waypoint_using_min_eibv,
#                                 self.ind_next_waypoint,
#                                 self.ind_current_waypoint,
#                                 self.ind_visited_waypoint,
#                                 filename=self.filename_ind_next)
#             self.ind_pioneer_waypoint = int(np.loadtxt(self.filename_ind_next))
#
#         # == plot gmrf section
#         if self.plot:
#             xrot = self.gmrf_grid[:, 0] * np.cos(ROTATED_ANGLE) - self.gmrf_grid[:, 1] * np.sin(ROTATED_ANGLE)
#             yrot = self.gmrf_grid[:, 0] * np.sin(ROTATED_ANGLE) + self.gmrf_grid[:, 1] * np.cos(ROTATED_ANGLE)
#             zrot = -self.gmrf_grid[:, 2]
#
#             ind_plot = np.where((zrot<0) * (zrot>=-5) * (xrot>70))[0]
#             mu_plot = self.knowledge.mu[ind_plot]
#             var_plot = self.knowledge.SigmaDiag[ind_plot]
#             ep_plot = get_excursion_prob(mu_plot.astype(np.float32), var_plot.astype(np.float32),
#                                          np.float32(self.gmrf_model.threshold))
#             self.yplot = xrot[ind_plot]
#             self.xplot = yrot[ind_plot]
#             self.zplot = zrot[ind_plot]
#
#             figpath = FIGPATH + self.agent_name + "/"
#             filename = figpath + "mean/jpg/P_{:03d}.jpg".format(step)
#             fig_mu = self.plot_figure(mu_plot, filename, vmin=10, vmax=27, opacity=.4, surface_count=6, cmap="BrBG",
#                                       cbar_title="Salinity", share=share, other_agent=other_agent, step=step,
#                                       ind_legal=ind_legal)
#
#             filename = figpath + "var/jpg/P_{:03d}.jpg".format(step)
#             fig_var = self.plot_figure(var_plot, filename, vmin=0, vmax=.1, opacity=.4, surface_count=4, cmap="Blues",
#                                        cbar_title="STD", share=share, other_agent=other_agent, reverse_scale=True,
#                                        step=step, ind_legal=ind_legal)
#
#             # filename = figpath + "ep/jpg/P_{:03d}.jpg".format(step)
#             # fig_ep = self.plot_figure(ep_plot, filename, vmin=0, vmax=1, opacity=.4, surface_count=10, cmap="Brwnyl",
#             #                           cbar_title="EP", other_agent=other_agent, share=share, step=step,
#             #                           ind_legal=ind_legal)
#
#             if step == NUM_STEPS - 1 or share:
#                 print("HTML is saved!")
#                 filename = figpath + "mean/html/P_{:03d}.html".format(step)
#                 wd = os.path.dirname(filename)
#                 checkfolder(wd)
#                 plotly.offline.plot(fig_mu, filename=filename, auto_open=False)
#
#                 filename = figpath + "var/html/P_{:03d}.html".format(step)
#                 wd = os.path.dirname(filename)
#                 checkfolder(wd)
#                 plotly.offline.plot(fig_var, filename=filename, auto_open=False)
#
#                 # filename = figpath + "ep/html/P_{:03d}.html".format(step)
#                 # wd = os.path.dirname(filename)
#                 # checkfolder(wd)
#                 # plotly.offline.plot(fig_ep, filename=filename, auto_open=False)
#
#         self.ind_previous_waypoint = self.ind_current_waypoint
#         self.ind_current_waypoint = self.ind_next_waypoint
#         self.ind_next_waypoint = self.ind_pioneer_waypoint
#         self.ind_visited_waypoint.append(self.ind_current_waypoint)
#         print("previous ind: ", self.ind_previous_waypoint)
#         print("current ind: ", self.ind_current_waypoint)
#         print("next ind: ", self.ind_next_waypoint)
#         print("pioneer ind: ", self.ind_pioneer_waypoint)
#
#     def save_agent_data(self):
#         datapath = FILEPATH + "AgentsData/" + self.agent_name + ".npy"
#         np.save(datapath, self.data_agent)
#         print("Data from " + self.agent_name + " is saved successfully!")
#
#     def load_data_from_other_agent(self, ag):
#         self.data_from_other_agent = np.empty([0, 2])
#         agent_name = ag.agent_name
#         datapath = FILEPATH + "AgentsData/" + agent_name + ".npy"
#         self.data_from_other_agent = np.load(datapath)
#         print("Data from " + agent_name + " is loaded successfully!")
#
#     def clear_agent_data(self):
#         self.data_agent = np.empty([0, 2])
#         print("Data from " + self.agent_name + " is cleared successfully!", self.data_agent)
#
#     def get_ind_sample(self, ind_start, ind_end):
#         N = 20
#         x_start, y_start, z_start = self.waypoints[ind_start, :]
#         x_end, y_end, z_end = self.waypoints[ind_end, :]
#         x_path = np.linspace(x_start, x_end, N)
#         y_path = np.linspace(y_start, y_end, N)
#         z_path = np.linspace(z_start, z_end, N)
#         dataset = np.vstack((x_path, y_path, z_path, np.zeros_like(z_path))).T
#         ind, value = self.assimilate_data(dataset)
#         return ind
#
#     def assimilate_data(self, dataset):
#         print("dataset before filtering: ", dataset[:10, :])
#         ind_remove_noise_layer = np.where(np.abs(dataset[:, 2]) >= .25)[0]
#         dataset = dataset[ind_remove_noise_layer, :]
#         print("dataset after filtering: ", dataset[:10, :])
#         t1 = time.time()
#         dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_gmrf_grid]) -
#               np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 0]).T) ** 2
#         dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_gmrf_grid]) -
#               np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 1]).T) ** 2
#         dz = ((vectorise(dataset[:, 2]) @ np.ones([1, self.N_gmrf_grid]) -
#               np.ones([dataset.shape[0], 1]) @ vectorise(self.gmrf_grid[:, 2]).T) * GMRF_DISTANCE_NEIGHBOUR) ** 2
#         dist = dx + dy + dz
#         ind_min_distance = np.argmin(dist, axis=1)
#         t2 = time.time()
#         ind_assimilated = np.unique(ind_min_distance)
#         salinity_assimilated = np.zeros(len(ind_assimilated))
#         for i in range(len(ind_assimilated)):
#             ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
#             salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
#         print("Data assimilation takes: ", t2 - t1)
#         self.auv_data = []
#         print("Reset auv_data: ", self.auv_data)
#         return ind_assimilated, vectorise(salinity_assimilated)
#
#     def plot_figure(self, value, filename, vmin=None, vmax=None, opacity=None, surface_count=None, cmap=None,
#                     cbar_title=None, share=False, other_agent=None, reverse_scale=False, step=0, ind_legal=None):
#         points_int, values_int = interpolate_3d(self.xplot, self.yplot, self.zplot, value)
#
#         # fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'},]])
#
#         # fig = go.Figure(data=go.Volume(
#         #     x=points_int[:, 0],
#         #     y=points_int[:, 1],
#         #     z=points_int[:, 2],
#         #     value=values_int.flatten(),
#         #     isomin=vmin,
#         #     isomax=vmax,
#         #     opacity=opacity,
#         #     surface_count=surface_count,
#         #     coloraxis="coloraxis",
#         #     caps=dict(x_show=False, y_show=False, z_show=False),
#         # ),
#         # )
#
#         # plot waypoint section first
#         xrot = self.waypoints[:, 0] * np.cos(ROTATED_ANGLE) - self.waypoints[:, 1] * np.sin(ROTATED_ANGLE)
#         yrot = self.waypoints[:, 0] * np.sin(ROTATED_ANGLE) + self.waypoints[:, 1] * np.cos(ROTATED_ANGLE)
#         zrot = -self.waypoints[:, 2]
#         fig = go.Figure(data=go.Scatter3d(name="Waypoint graph",
#             x=yrot,
#             y=xrot,
#             z=zrot,
#             mode='markers',
#             marker=dict(color='black', size=1, opacity=.1)
#         ))
#
#         depth_layer = np.unique(points_int[:, 2])
#         for i in range(len(depth_layer)):
#             ind_depth = np.where(points_int[:, 2] == depth_layer[i])[0]
#             fig.add_trace(go.Isosurface(name="Salinity field at depth" + str(depth_layer[i]),
#                                         x=points_int[ind_depth, 0],
#                                         y=points_int[ind_depth, 1],
#                                         z=points_int[ind_depth, 2],
#                                         value=values_int[ind_depth].flatten(),
#                                         isomin=vmin,
#                                         isomax=vmax,
#                                         # opacity=opacity,
#                                         # surface_count=surface_count,
#                                         coloraxis="coloraxis",
#                                         # caps=dict(x_show=False, y_show=False, z_show=False),
#                                         ),
#                           )
#             # ind_depth = np.where(points_int[:, 2] == depth_layer[i])[0]
#             # fig.add_trace(go.Isosurface(name="Salinity field at depth" + str(depth_layer[i]),
#             #     x=points_int[ind_depth, 0],
#             #     y=points_int[ind_depth, 1],
#             #     z=points_int[ind_depth, 2],
#             #     value=values_int[ind_depth].flatten(),
#             #     isomin=vmin,
#             #     isomax=vmax,
#             #     opacity=opacity,
#             #     # surface_count=surface_count,
#             #     coloraxis="coloraxis",
#             #     caps=dict(x_show=False, y_show=False, z_show=False),
#             # ),
#             # )
#
#         # == plot waypoint section
#         # xrot = self.waypoints[:, 0] * np.cos(ROTATED_ANGLE) - self.waypoints[:, 1] * np.sin(ROTATED_ANGLE)
#         # yrot = self.waypoints[:, 0] * np.sin(ROTATED_ANGLE) + self.waypoints[:, 1] * np.cos(ROTATED_ANGLE)
#         # zrot = -self.waypoints[:, 2]
#         # fig.add_trace(go.Scatter3d(name="Waypoint graph",
#         #     x=yrot,
#         #     y=xrot,
#         #     z=zrot,
#         #     mode='markers',
#         #     marker=dict(color='black', size=1, opacity=.1)
#         # ))
#         fig.add_trace(go.Scatter3d(name="Previous waypoint",
#             x=[yrot[self.ind_previous_waypoint]],
#             y=[xrot[self.ind_previous_waypoint]],
#             z=[zrot[self.ind_previous_waypoint]],
#             mode='markers',
#             marker=dict(color='yellow', size=10)
#         ))
#         fig.add_trace(go.Scatter3d(name="Current waypoint",
#             x=[yrot[self.ind_current_waypoint]],
#             y=[xrot[self.ind_current_waypoint]],
#             z=[zrot[self.ind_current_waypoint]],
#             mode='markers',
#             marker=dict(color='red', size=10)
#         ))
#         fig.add_trace(go.Scatter3d(name="Next waypoint",
#             x=[yrot[self.ind_next_waypoint]],
#             y=[xrot[self.ind_next_waypoint]],
#             z=[zrot[self.ind_next_waypoint]],
#             mode='markers',
#             marker=dict(color='blue', size=10)
#         ))
#         fig.add_trace(go.Scatter3d(name="Pioneer waypoint",
#             x=[yrot[self.ind_pioneer_waypoint]],
#             y=[xrot[self.ind_pioneer_waypoint]],
#             z=[zrot[self.ind_pioneer_waypoint]],
#             mode='markers',
#             marker=dict(color='green', size=10)
#         ))
#         fig.add_trace(go.Scatter3d(name="Visited waypoints",
#             x=yrot[self.ind_visited_waypoint],
#             y=xrot[self.ind_visited_waypoint],
#             z=zrot[self.ind_visited_waypoint],
#             mode='markers+lines',
#             marker=dict(color='black', size=4),
#             line=dict(color='black', width=3)
#         ))
#         if ind_legal is not None:
#             fig.add_trace(go.Scatter3d(name="Legal waypoints",
#                 x=yrot[ind_legal],
#                 y=xrot[ind_legal],
#                 z=zrot[ind_legal],
#                 mode='markers',
#                 marker=dict(color='black', size=5, opacity=.3),
#                 # line=dict(color='black', width=3)
#             ))
#
#         # plot other agent
#         if other_agent:
#             fig.add_trace(go.Scatter3d(name=other_agent.agent_name + "\'s current waypoint",
#                 x=[yrot[other_agent.ind_current_waypoint]],
#                 y=[xrot[other_agent.ind_current_waypoint]],
#                 z=[zrot[other_agent.ind_current_waypoint]],
#                 mode='markers',
#                 marker=dict(color='brown', size=5, opacity=1)
#             ))
#             fig.add_trace(go.Scatter3d(name=other_agent.agent_name + "\'s trajectory",
#                 x=yrot[other_agent.ind_visited_waypoint],
#                 y=xrot[other_agent.ind_visited_waypoint],
#                 z=zrot[other_agent.ind_visited_waypoint],
#                 mode='markers+lines',
#                 marker=dict(color='grey', size=5),
#                 line=dict(color='grey', width=3),
#                 opacity=1,
#             ))
#
#         if share:
#             ind_measured_by_other_agent = self.ind_measured_by_other_agent.astype(int)
#             xrot = (self.gmrf_grid[ind_measured_by_other_agent, 0] * np.cos(ROTATED_ANGLE) -
#                     self.gmrf_grid[ind_measured_by_other_agent, 1] * np.sin(ROTATED_ANGLE))
#             yrot = (self.gmrf_grid[ind_measured_by_other_agent, 0] * np.sin(ROTATED_ANGLE) +
#                     self.gmrf_grid[ind_measured_by_other_agent, 1] * np.cos(ROTATED_ANGLE))
#             zrot = - self.gmrf_grid[ind_measured_by_other_agent, 2]
#
#             fig.add_trace(go.Scatter3d(name="Shared waypoints",
#                                        x=yrot,
#                                        y=xrot,
#                                        z=zrot,
#                                        mode='markers',
#                                        marker=dict(color='cyan', size=4,
#                                                    opacity=.5),
#                                        ))
#         # fig.add_trace(go.Scatter3d(
#         #     x=yrot[self.myopic3d_planner.ind_candidates],
#         #     y=xrot[self.myopic3d_planner.ind_candidates],
#         #     z=zrot[self.myopic3d_planner.ind_candidates],
#         #     mode='markers',
#         #     marker=dict(color='orange', size=5, opacity=.3)
#         # ))
#         fig.update_coloraxes(colorscale=cmap, colorbar=dict(lenmode='fraction', len=.5, thickness=20,
#                                                               tickfont=dict(size=18, family="Times New Roman"),
#                                                               title="Salinity",
#                                                               titlefont=dict(size=18, family="Times New Roman")),
#                              colorbar_title=cbar_title,
#                              reversescale=reverse_scale)
#
#         def rotate_z(x, y, z, theta):
#             w = x + 1j * y
#             return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z
#
#         xe, ye, ze = [-1.25, -1.25, .5]
#         xe, ye, ze = rotate_z(xe, ye, ze, -step * .1)
#         camera = dict(
#             up=dict(x=0, y=0, z=1),
#             center=dict(x=0, y=0, z=0),
#             # eye=dict(x=-1.25, y=-1.25, z=.5)
#             eye=dict(x=xe, y=ye, z=ze)
#         )
#         fig.update_layout(coloraxis_colorbar_x=0.8)
#         fig.update_layout(
#             title={
#                 'text': "Adaptive 3D myopic illustration",
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top',
#                 'font': dict(size=30, family="Times New Roman"),
#             },
#             scene=dict(
#                 zaxis=dict(nticks=4, range=[-5.5, -.5], ),
#                 xaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 yaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 zaxis_tickfont=dict(size=14, family="Times New Roman"),
#                 xaxis_title=dict(text="Y", font=dict(size=18, family="Times New Roman")),
#                 yaxis_title=dict(text="X", font=dict(size=18, family="Times New Roman")),
#                 zaxis_title=dict(text="Z", font=dict(size=18, family="Times New Roman")),
#             ),
#             scene_aspectmode='manual',
#             scene_aspectratio=dict(x=1, y=1, z=.5),
#             scene_camera=camera,
#         )
#
#         # plotly.offline.plot(fig, filename=FIGPATH + "myopic3d/P_{:03d}.html".format(i), auto_open=False)
#         wd = os.path.dirname(filename)
#         checkfolder(wd)
#         # if not os.path.exists(wd):
#         #     pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
#         #     print(wd + " is created successfully!")
#         fig.write_image(filename, width=1980, height=1080)
#         return fig
#
#     def check_agent(self):
#         ag1_loc = [63.451022, 10.396262, .5]
#         ag2_loc = [63.452381, 10.424680, .5]
#         self.plot = True
#         self.prepare_run(ag1_loc, ind_legal=np.arange(self.waypoints.shape[0]))
#         # for i in range(20):
#         #     self.sample()
#         #     self.run(i)
#         # self.set_starting_location(AGENT1_START_LOCATION)
#         # self.prepare_run()
#         # self.run()
#         # self.monitor_data()
#         pass
#
#
# if __name__ == "__main__":
#     a = Agent()
#     # NUM_STEPS = 1
#     a.check_agent()
#
# # #%%
# # plt.plot(a.waypoints[:, 1], a.waypoints[:, 0], 'k.')
# # plt.show()
#
#
#
