"""
This script contains essential functions for plotting
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-17
"""

import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10), cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None, stepsize=None, threshold=None, xlabel=None, ylabel=None):

    triang = tri.Triangulation(x, y)
    refiner = tri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(values.flatten(), subdiv=3)

    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if np.any([vmin, vmax]):
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None

    if np.any(levels):
        linewidths = np.ones_like(levels) * .3
        colors = len(levels) * ['black']
        if threshold:
            dist = np.abs(threshold - levels)
            ind = np.where(dist == np.amin(dist))[0]
            linewidths[ind] = 3
            colors[ind[0]] = 'red'
        contourplot = ax.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap, alpha=alpha)
        ax.tricontour(tri_refi, z_test_refi, levels=levels, linewidths=linewidths, colors=colors,
                      alpha=alpha)
    else:
        contourplot = ax.tricontourf(tri_refi, z_test_refi, cmap=cmap, alpha=alpha)
        ax.tricontour(tri_refi, z_test_refi, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# x = a.waypoints[:, 0]
# y = a.waypoints[:, 1]
#
# z = a.grf_model.mu_truth
# plotf_vector(x, y, z, vmin=0, vmax=1.2, stepsize=.1, threshold=.7)
# plt.show()




