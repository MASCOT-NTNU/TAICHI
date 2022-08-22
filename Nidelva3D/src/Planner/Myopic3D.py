"""

"""
from Planner.Planner import Planner
from WaypointGraph import WaypointGraph


class Myopic3D(Planner):

    def __init__(self, wp: WaypointGraph) -> None:
        super().__init__()
        self.wp = wp

    def get_candidates(self):
        id_n = self.wp.get_ind_neighbours(self._id_now)
        wp_now = self.wp.get_waypoint_from_ind(self._id_now)
        id_smooth = []
        for ids in id_n:
            wps = self.wp.get_waypoint_from_ind(ids)

            pass

    def get_vec_from_two_waypoints(self):
        pass

    def get_next_waypoint(self):
        pass

#%%



#%%
# """
# This script determines the next index based on minimum EIBV
# Author: Yaolin Ge
# Contact: yaolin.ge@ntnu.no
# Date: 2022-07-25
# """
#
#
# from Planner import Planner
#
#
# class Myopic3D(Planner):
#
#     def __init__(self):
#         pass
#
#     def plan_one_step_ahead(self):
#         pass
#
#
#
#
# import time
# import pickle
#
#
# vectorize(['float32(float32, float32, float32)'], target='cuda')
# def get_eibv_from_gpu(mu, SigmaDiag, threshold):
#   cdf = norm.cdf(threshold, mu, SigmaDiag)
#   bv = cdf*(1-cdf)
#   ibv = np.sum(bv)
#   return ibv
#
#
# def get_eibv_from_fast(mu, sigma, threshold):
#   p = norm.cdf(threshold, mu, sigma)
#   bv = p * (1 - p)
#   ibv = np.sum(bv)
#   return ibv
#
#
# class MyopicPlanning3D:
#
#     def __init__(self, waypoints=None, hash_neighbours=None, hash_waypoint2gmrf=None):
#         self.waypoints = waypoints
#         self.hash_neighbours = hash_neighbours
#         self.hash_waypoint2gmrf = hash_waypoint2gmrf
#         print("MyopicPlanner is ready")
#
#     def update_planner(self, knowledge=None, gmrf_model=None, ind_legal=None):
#         self.knowledge = knowledge
#         self.gmrf_model = gmrf_model
#         if ind_legal is None:
#             ind_legal = []
#         self.ind_legal=ind_legal
#         print("Planner is updated successfully!")
#
#     def find_next_waypoint_using_min_eibv(self, ind_current=None, ind_previous=None, ind_visited=None, filename=None):
#         self.ind_current = ind_current
#         self.ind_previous = ind_previous
#         self.ind_visited = ind_visited
#
#         self.find_all_neighbours()
#         self.smooth_filter_neighbours()
#         t1 = time.time()
#
#         print("updateEIBV: ", self.get_eibv_from_gmrf_model(vectorise(self.ind_candidates)))
#         self.EIBV = self.get_eibv_from_gmrf_model(vectorise(self.ind_candidates))
#         if len(self.EIBV) > 0:
#             self.ind_next = self.ind_candidates[np.argmin(self.EIBV)]
#         else:
#             self.ind_next = self.ind_neighbours[np.random.randint(len(self.ind_neighbours))]
#
#         t2 = time.time()
#         print("Path planning takes: ", t2 - t1)
#         np.savetxt(filename, np.array([self.ind_next]))
#         print("ind_next is saved!")
#         return self.ind_next
#
#     def find_all_neighbours(self):
#         self.ind_neighbours = self.hash_neighbours[self.ind_current]
#         self.ind_neighbours = list(set(self.ind_neighbours).intersection(self.ind_legal))
#         print("legal neighbours: ", self.ind_neighbours)
#
#     def smooth_filter_neighbours(self):
#         vec1 = self.get_vec_from_indices(self.ind_previous, self.ind_current)
#         self.ind_candidates = []
#         for i in range(len(self.ind_neighbours)):
#             ind_candidate = self.ind_neighbours[i]
#             if not ind_candidate in self.ind_visited:
#                 vec2 = self.get_vec_from_indices(self.ind_current, ind_candidate)
#                 if np.dot(vec1.T, vec2) >= 0:
#                     self.ind_candidates.append(ind_candidate)
#
#     def get_vec_from_indices(self, ind_start, ind_end):
#         x_start = self.waypoints[ind_start, 0]
#         y_start = self.waypoints[ind_start, 1]
#         z_start = self.waypoints[ind_start, 2]
#
#         x_end = self.waypoints[ind_end, 0]
#         y_end = self.waypoints[ind_end, 1]
#         z_end = self.waypoints[ind_end, 2]
#
#         dx = x_end - x_start
#         dy = y_end - y_start
#         dz = z_end - z_start
#
#         return vectorise([dx, dy, dz])
#
#     def get_eibv_from_gmrf_model(self, ind_candidate):
#         t1 = time.time()
#         variance_post = self.gmrf_model.candidate(ks=ind_candidate)  # update the field
#         t2 = time.time()
#         # print("Update variance take: ", t2 - t1)
#
#         # eibv = get_eibv_from_gpu(self.knowledge.mu, variance_post)
#         N_eibv = variance_post.shape[1]
#         eibv = np.zeros(N_eibv)
#         t1 = time.time()
#         for i in range(N_eibv):
#             eibv[i] = get_eibv_from_fast(self.knowledge.mu, variance_post[:, i], self.gmrf_model.threshold)
#         t2 = time.time()
#         # print("EIBV calculation takes: ", t2 - t1)
#         return eibv
#
#     def check_multiprocessing(self):
#         waypoints = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
#
#         neighbour_file = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
#         hash_neighbours = pickle.load(neighbour_file)
#         neighbour_file.close()
#
#         waypoint2gmrf_file = open(FILEPATH + "Config/HashWaypoint2GMRF.p", 'rb')
#         hash_waypoint2gmrf = pickle.load(waypoint2gmrf_file)
#         waypoint2gmrf_file.close()
#         self.myopic3d_planner = MyopicPlanning3D(waypoints=waypoints, hash_neighbours=hash_neighbours,
#                                             hash_waypoint2gmrf=hash_waypoint2gmrf)
#
#         gmrf_grid = pd.read_csv(FILEPATH + "Config/GMRFGrid.csv").to_numpy()
#         N_gmrf_grid = len(gmrf_grid)
#
#         self.gmrf_model = spde(model=2, reduce=True, method=2)
#
#         self.knowledge = Knowledge(gmrf_grid=gmrf_grid, mu=self.gmrf_model.mu, SigmaDiag=self.gmrf_model.mvar())
#         self.myopic3d_planner.update_planner(knowledge=self.knowledge, gmrf_model=self.gmrf_model)
#
#         self.myopic3d_planner.find_next_waypoint_using_min_eibv(ind_current=0, ind_previous=1, ind_visited=[])
#         print("Finished")
#         pass
#
#
# if __name__ == "__main__":
#     my = MyopicPlanning3D()
#     my.check_multiprocessing()
#
#
#
#
#
#
#
#
#
