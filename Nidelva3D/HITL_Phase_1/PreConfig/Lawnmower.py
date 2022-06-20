"""
This script generates the lawnmower pattern
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-20
"""
import pandas as pd
import plotly.offline

from TAICHI.Nidelva3D.Config.Config import *
from usr_func import *


class LawnMowerPlanning:

    def __init__(self, starting_loc=None, ending_loc=None, no_legs=5, max_pitch=10, max_depth=5.5, min_depth=.5):
        self.starting_loc = starting_loc
        self.ending_loc = ending_loc
        self.no_legs = no_legs
        self.max_pitch = max_pitch
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.load_polygon_border()
        self.load_boundary()
        self.design_yoyo_lawnmower()
        pass

    def load_polygon_border(self):
        self.polygon_border = pd.read_csv(FILEPATH + "PreConfig/polygon_border.csv").to_numpy()
        self.polygon_border_shapely = Polygon(self.polygon_border)
        print("Polygon border is loaded successfully!")

    def load_boundary(self):
        self.boundary = np.load(FILEPATH + "models/grid.npy")

        lat = self.boundary[:, 2]
        lon = self.boundary[:, 3]
        x, y = latlon2xy(lat, lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)

        xp = x * np.cos(ROTATED_ANGLE) - y * np.sin(ROTATED_ANGLE)
        yp = x * np.sin(ROTATED_ANGLE) + y * np.cos(ROTATED_ANGLE)

        self.xmin = np.amin(xp)
        self.ymin = np.amin(yp)
        self.xmax = np.amax(xp)
        self.ymax = np.amax(yp)

    def design_yoyo_lawnmower(self):
        self.yoyo_lateral_distance = (self.max_depth - self.min_depth) / np.tan(deg2rad(self.max_pitch))
        self.x_yoyo = np.arange(self.xmin, self.xmax, self.yoyo_lateral_distance)
        self.y_yoyo = np.arange(self.ymin + 1, self.ymax, self.yoyo_lateral_distance)

        xx, yy = np.meshgrid(self.x_yoyo, self.y_yoyo)
        ylegs = np.linspace(self.ymin, self.ymax, self.no_legs)
        stepsize = ylegs[1] - ylegs[0]
        self.lawnmower_trajectory = []

        ind = []
        for i in range(len(ylegs)):
            ind.append(np.argmin((self.y_yoyo - ylegs[i])**2))  # check where to turn

        counter = 0
        for j in range(len(self.y_yoyo)):
            if j in ind:
                if isEven(counter):
                    for i in range(len(self.x_yoyo)):
                        x_temp, y_temp = self.x_yoyo[i], self.y_yoyo[j]
                        xn = x_temp * np.cos(ROTATED_ANGLE) + y_temp * np.sin(ROTATED_ANGLE)
                        yn = -x_temp * np.sin(ROTATED_ANGLE) + y_temp * np.cos(ROTATED_ANGLE)
                        point = Point(xn, yn)
                        if self.polygon_border_shapely.contains(point):
                            self.lawnmower_trajectory.append([xn, yn])
                    counter += 1
                else:
                    for i in range(len(self.x_yoyo)-1, -1, -1):
                        x_temp, y_temp = self.x_yoyo[i], self.y_yoyo[j]
                        xn = x_temp * np.cos(ROTATED_ANGLE) + y_temp * np.sin(ROTATED_ANGLE)
                        yn = -x_temp * np.sin(ROTATED_ANGLE) + y_temp * np.cos(ROTATED_ANGLE)
                        point = Point(xn, yn)
                        if self.polygon_border_shapely.contains(point):
                            self.lawnmower_trajectory.append([xn, yn])
                    counter += 1
            else:
                x_temp, y_temp = self.x_yoyo[i], self.y_yoyo[j]
                xn = x_temp * np.cos(ROTATED_ANGLE) + y_temp * np.sin(ROTATED_ANGLE)
                yn = -x_temp * np.sin(ROTATED_ANGLE) + y_temp * np.cos(ROTATED_ANGLE)
                if self.polygon_border_shapely.contains(point):
                    self.lawnmower_trajectory.append([xn, yn])

        path = np.array(self.lawnmower_trajectory)
        dist = []
        for i in range(len(path) - 1):
            dist_x = path[i, 0] - path[i+1, 0]
            dist_y = path[i, 1] - path[i+1, 1]
            dist.append(np.sqrt(dist_x ** 2 + dist_y ** 2))
        ind = np.where(dist > stepsize)[0][0] # check singularity

        x1, y1 = self.lawnmower_trajectory.pop(ind)
        x2, y2 = self.lawnmower_trajectory.pop(ind)
        x, y = self.distribute_along_path(x1, y1, x2, y2, self.yoyo_lateral_distance)
        for i in range(len(x)):
            print([x[i], y[i]])
            self.lawnmower_trajectory.insert(ind, [x[i], y[i]])
            ind += 1

        self.lawnmower_trajectory_3d = []
        for i in range(len(self.lawnmower_trajectory)):
            if isEven(i):
                lat, lon = xy2latlon(self.lawnmower_trajectory[i][0], self.lawnmower_trajectory[i][1],
                                     LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                self.lawnmower_trajectory_3d.append([lat, lon, self.min_depth])
            else:
                self.lawnmower_trajectory_3d.append([lat, lon, self.max_depth])

        df = pd.DataFrame(self.lawnmower_trajectory_3d, columns=['lat', 'lon', 'depth'])
        df.to_csv(FILEPATH + "Config/lawnmower.csv", index=False)
        print("Path is saved successfully!")

    def distribute_along_path(self, x1, y1, x2, y2, stepsize):
        alpha = np.math.atan2(x2 - x1, y2 - y1)
        xgap = stepsize * np.sin(alpha)
        ygap = stepsize * np.cos(alpha)
        x = np.arange(x1, x2, xgap)
        y = np.arange(y1, y2, ygap)
        print("xgap: ", xgap)
        print("ygap: ", ygap)
        return x, y

    def get_distance_of_trajectory(self):
        dist = 0
        path = np.array(self.lawnmower_trajectory_3d)
        for i in range(len(path) - 1):
            dist_x = path[i, 0] - path[i+1, 0]
            dist_y = path[i, 1] - path[i+1, 1]
            dist_z = path[i, 2] - path[i+1, 2]
            dist += np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z**2)
        print("Distance of the trajectory: ", dist)
        return dist

    def check_path(self):
        traj = np.array(self.lawnmower_trajectory)
        # plt.plot(yy, xx, 'k.')
        # for i in range(ylegs.shape[0]):
        #     plt.axvline(ylegs[i])
        plt.plot(traj[:, 1], traj[:, 0], 'g.-')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.show()

        trj = np.array(self.lawnmower_trajectory_3d)
        x = trj[:, 1]
        y = trj[:, 0]
        z = -trj[:, 2]

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers + lines',
            marker=dict(
                size=12,
            ),
            line=dict(
                width=2,
            )
        )])

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        plotly.offline.plot(fig, filename=FIGPATH + "lawnmower.html", auto_open=True)
    pass


if __name__ == "__main__":
    starting_loc =  [1000, -1100]
    ending_loc = [1000, 1000]
    l = LawnMowerPlanning(starting_loc=starting_loc, ending_loc=ending_loc, no_legs=4)
    l.check_path()






