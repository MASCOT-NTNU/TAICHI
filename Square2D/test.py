import numpy as np
import scipy.spatial.distance as scdist
import matplotlib.pyplot as plt
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
xx, yy = np.meshgrid(x, y)
X = xx.reshape(-1, 1)
Y = yy.reshape(-1, 1)
grid = np.hstack((X, Y))

sigma = 1
eta = 10
t = scdist.cdist(grid, grid)
Sigma = sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)
L = np.linalg.cholesky(Sigma)
z = np.random.randn(len(X)).reshape(-1, 1)
mu = L @ z
plt.imshow(mu.reshape(50, 50), cmap = "RdBu")
plt.xlabel("s1")
plt.ylabel("s2")
plt.title("Realisation of the GRF")
plt.colorbar()

figpath = "/Users/yaolin/HomeOffice/TAICHI/Square2D/fig/GP"
plt.savefig(figpath + "3.pdf")
plt.show()
#%%
x = np.arange(10)
y = np.arange(10)
xx, yy = np.meshgrid(x, y)
path_x = [0, 1, 1, 2, 3, 4]
path_y = [0, 1, 2, 2, 1, 2]
plt.figure(figsize=(5, 5))
current_x = path_x[-1]
current_y = path_y[-1]
#
line1x = [current_x - 1, current_x + 1]
line1y = [current_y - 1, current_y + 1]
plt.plot(line1x, line1y, "g-")

line1x = [current_x - 1, current_x + 1]
line1y = [current_y + 1, current_y - 1]
plt.plot(line1x, line1y, "g-")

line1x = [current_x, current_x]
line1y = [current_y + 1, current_y - 1]
plt.plot(line1x, line1y, "g-")

line1x = [current_x - 1, current_x + 1]
line1y = [current_y, current_y]
plt.plot(line1x, line1y, "g-")
#
line1x = [current_x - 1, current_x]
line1y = [current_y, current_y]
plt.plot(line1x, line1y, "r-")

line1x = [current_x, current_x]
line1y = [current_y - 1, current_y]
plt.plot(line1x, line1y, "r-")



X = xx.reshape(-1, 1)
Y = yy.reshape(-1, 1)
plt.plot(X, Y, 'k.')
plt.plot(path_x, path_y, "b-")

line1x = [current_x - 1, current_x]
line1y = [current_y - 1, current_y]
plt.plot(line1x, line1y, "r-")

plt.xlabel("s1")
plt.ylabel("s2")
plt.title("Waypoint graph")
plt.savefig(figpath + "way3.pdf")
plt.show()

#%%


def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)


def mu(H, beta):
    '''
    :param H: design matrix
    :param beta: regression coef
    :return: mean
    '''
    return np.dot(H, beta)

import matplotlib.pyplot as plt

print("hello world")
from usr_func import *
figpath = "/Users/yaolin/HomeOffice/TAICHI/Square2D/fig/GP/Simulation/"
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Setup the grid
n1 = 25 # number of grid points along east direction, or x, or number of columns
n2 = 25 # number of grid points along north direction, or y, or number of rows
n = n1 * n2 # total number of  grid points

XLIM = [0, 1] # limits in the grid
YLIM = [0, 1]

sites1 = np.linspace(XLIM[0], XLIM[1], n1)
sites2 = np.linspace(XLIM[0], YLIM[1], n2)
sites1m, sites2m = np.meshgrid(sites1, sites2)
sites1v = sites1m.reshape(-1, 1) # sites1v is the vectorised version
sites2v = sites2m.reshape(-1, 1)

# Compute the distance matrix
grid = np.hstack((sites1v, sites2v))
t = scdist.cdist(grid, grid)

# Simulate the initial random field
# alpha = 1.0 # beta as in regression model
sigma = .5  # scaling coef in matern kernel
tau = .01 # iid noise
eta = 5 # coef in matern kernel

# only one parameter is considered for salinity
beta = [[29.0], [.01], [0.01]] # [intercept, trend along east and north

Sigma = Matern_cov(sigma, eta, t)  # matern covariance
plt.imshow(Sigma)
plt.show()

L = np.linalg.cholesky(Sigma)  # lower triangle covariance matrix
z = np.dot(L, np.random.randn(n).reshape(-1, 1)) # sampled randomly with covariance structure

# generate the prior of the field
H = np.hstack((np.ones([n, 1]), sites1v, sites2v)) # different notation for the project
mu_prior = mu(H, beta).reshape(n, 1)
plt.imshow(mu_prior.reshape(n1, n2))
plt.colorbar()
plt.show()
mu_real = mu_prior + z  # add covariance structured noise
S_thres = np.mean(mu_real)
plt.imshow(mu_real.reshape(n1, n2))
plt.colorbar()
plt.show()

def EP_1D(mu, Sig, Thres):
    '''
    :param mu:
    :param Sig:
    :param T_thres:
    :param S_thres:
    :return:
    '''
    n = mu.shape[0]
    ES_Prob = np.zeros([n, 1])
    for i in range(n):
        ES_Prob[i] = norm.cdf(Thres, mu[i], Sig[i, i])
    return ES_Prob

EP = EP_1D(mu_real, Sigma, S_thres)
plt.imshow(EP.reshape(n1, n2), cmap = "RdBu", interpolation="spline36", vmin = 0, vmax = 1)
plt.xlabel("s1")
plt.ylabel("s2")
plt.title("Excursion probability on the true field")
plt.colorbar()
plt.savefig(figpath + "true.pdf")
plt.show()

#%%
plt.imshow(mu_real.reshape(n1, n2), cmap = "YlGnBu", interpolation="spline36")
plt.xlabel("s1")
plt.ylabel("s2")
plt.title("True field")
plt.colorbar()
plt.savefig(figpath + "True.pdf")
plt.show()

#%%

def compute_ES(mu, Thres):
    '''
    :param mu:
    :param Tthres:
    :return:
    '''
    excursion = np.copy(mu)

    excursion[mu > Thres] = 0
    excursion[mu < Thres] = 1

    return excursion

#% Method I: functions used for path planning purposes

def find_starting_loc(ep, n1, n2):
    '''
    This will find the starting location in
    the grid according to the excursion probability
    which is closest to 0.5
    :param ep:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(ep - ep_criterion)).argmin()
    row_ind, col_ind = np.unravel_index(ind, (n2, n1))
    return row_ind, col_ind



def find_neighbouring_loc(row_ind, col_ind):
    '''
    This will find the neighbouring loc
    But also limit it inside the grid
    :param idx:
    :param idy:
    :return:
    '''

    row_ind_l = [row_ind - 1 if row_ind > 0 else row_ind]
    row_ind_u = [row_ind + 1 if row_ind < n2 - 1 else row_ind]
    col_ind_l = [col_ind - 1 if col_ind > 0 else col_ind]
    col_ind_u = [col_ind + 1 if col_ind < n1 - 1 else col_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))

    row_ind, col_ind = np.meshgrid(row_ind_v, col_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)


def find_next_EIBV(row_neighbour, col_neighbour, row_now, col_now, Sig, mu, tau, S_thres):

    id = []

    for i in row_neighbour:
        for j in col_neighbour:
            if i == row_now and j == col_now:
                continue
            id.append(np.ravel_multi_index((i, j), (n2, n1)))
    id = np.unique(np.array(id))

    M = len(id)
    R = tau ** 2

    eibv = []
    for k in range(M):
        F = np.zeros([1, n])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(S_thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next = np.unravel_index(id[ind_desired], (n2, n1))
    return row_next, col_next, id, eibv


def GPupd(mu, Sig, R, F, y_sampled):
    C = np.dot(F, np.dot(Sig, F.T)) + R
    mu_p = mu + np.dot(Sig, np.dot(F.T, np.linalg.solve(C, (y_sampled - np.dot(F, mu)))))
    Sigma_p = Sig - np.dot(Sig, np.dot(F.T, np.linalg.solve(C, np.dot(F, Sig))))
    return mu_p, Sigma_p


#%% Static design
# move a vertical line
M = n2
mu_cond = mu_prior
Sigma_cond = Sigma
path_row = []
path_col = []
figpath = "/Users/yaolin/HomeOffice/TAICHI/Square2D/fig/GP/Simulation/Static/"
# if not os.path.exists(figpath + "Static/"):
#     os.mkdir(figpath + "Static/")
distance = 41
speed = 1.5
tid = []
accuracy = []

rdbu = cm.get_cmap('RdBu', 256)
newcolors = rdbu(np.linspace(0, 1, 35))
newcmp = ListedColormap(newcolors)


for j in range(M):
    print(j)
    k1 = j % n2
    k2 = int(j / n2)
    F = np.zeros([1, n])
    F[0, np.ravel_multi_index((j, n2 - j - 1), (n1, n2))] = True # select col 13 to move along
    path_row.append(j) # only for plotting
    path_col.append(n2 - j - 1)


    # switch = False
    # if k2 % 2 == 0:
    #     if j != 0:
    #         switch = True
    #     F[0, np.ravel_multi_index((k1, 12 * k2), (n1, n2))] = True # select col 13 to move along
    #     path_row.append(k1) # only for plotting
    #     path_col.append(12 * k2)
    # else:
    #     switch = True
    #     F[0, np.ravel_multi_index((n1 - k1 - 1, 12 * k2), (n1, n2))] = True  # select col 13 to move along
    #     path_row.append(n1 - k1 - 1) # only for plotting
    #     path_col.append(12 * k2)

    R = np.diagflat(tau ** 2)
    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1)

    mu_cond, Sigma_cond = GPupd(mu_cond, Sigma_cond, R, F, y_sampled)
    accuracy.append(np.mean((mu_cond - mu_real) ** 2))
    # if switch == True:
    #     tid.append((distance / speed) * j * 5)
    # else:
    #     tid.append((distance / speed) * j)
    ES_Prob = EP_1D(mu_cond, Sigma_cond, S_thres)
    ES_Prob_m = np.array(ES_Prob).reshape(n1, n2) # ES_Prob_m is the reshaped matrix form of ES_Prob

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(nrows=2, ncols=2)

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin=0, vmax=1, cmap = newcmp, interpolation="spline16");
    plt.title("Excursion probability on the field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(mu_cond.reshape(n2, n1), cmap = newcmp, interpolation="spline16");
    plt.title("Realised field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.sqrt(np.diag(Sigma_cond)).reshape(n2, n1), cmap = "binary");
    plt.title("Uncertainty")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");
    plt.ylabel("s2");
    plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    plt.title("Mean Squared Error")
    plt.plot(accuracy, 'k.-', linewidth = 2)
    plt.xlabel("Iterations")

    plt.savefig(figpath + "S_{:03d}.png".format(j))
    plt.close("all")

#%% Method III: Functions used for rule-based system
'''
In this section, the rule-based system is implemented to do the filtering 
procedure on the candidate grid nodes where unrealistic nodes are eliminated 
using cross-product rule, which only allows the AUV to run with a certain operational 
angle limits 
'''


def ExpectedVarianceUsr(threshold, mu, Sig, F, R):
    '''
    :param threshold:
    :param mu:
    :param Sig:
    :param F: sampling matrix
    :param R: noise matrix
    :return:
    '''
    Sigxi = np.dot(Sig, np.dot(F.T, np.linalg.solve(np.dot(F, np.dot(Sig, F.T)) + R, np.dot(F, Sig))))
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1) # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        sn = np.sqrt(sn2) # the corresponding standard deviation term
        m = mu[i]
        # mur = (threshold - m) / sn
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2

    return IntA


def find_starting_loc_rule(ep):
    '''
    This will find the starting location in
    the grid according to the excursion probability
    which is closest to 0.5
    :param ep:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(ep - ep_criterion)).argmin()
    row_ind, col_ind = np.unravel_index((ind), (n2, n1))
    return row_ind, col_ind


def find_candidates_loc_rule(row_ind, col_ind):
    '''
    This will find the neighbouring loc
    But also limit it inside the grid
    :param idx:
    :param idy:
    :return:
    '''

    row_ind_l = [row_ind - 1 if row_ind > 0 else row_ind]
    row_ind_u = [row_ind + 1 if row_ind < n2 - 1 else row_ind]
    col_ind_l = [col_ind - 1 if col_ind > 0 else col_ind]
    col_ind_u = [col_ind + 1 if col_ind < n1 - 1 else col_ind]

    row_ind_v = np.unique(np.vstack((row_ind_l, row_ind, row_ind_u)))
    col_ind_v = np.unique(np.vstack((col_ind_l, col_ind, col_ind_u)))

    row_ind, col_ind = np.meshgrid(row_ind_v, col_ind_v)

    return row_ind.reshape(-1, 1), col_ind.reshape(-1, 1)


def find_next_EIBV_rule(row_cand, col_cand, row_now, col_now, row_previous, col_previous, Sig, mu, tau, Thres):

    id = []
    drow1 = row_now - row_previous
    dcol1 = col_now - col_previous
    vec1 = np.array([dcol1, drow1])
    for i in row_cand:
        for j in col_cand:
            if i == row_now and j == col_now:
                continue
            drow2 = i - row_now
            dcol2 = j - col_now
            vec2 = np.array([dcol2, drow2])
            if np.dot(vec1, vec2) >= 0: # add the rule for not turning sharply
                id.append(np.ravel_multi_index((i, j), (n2, n1)))
            else:
                continue
    id = np.unique(np.array(id))

    M = len(id)
    R = tau ** 2

    eibv = []
    for k in range(M):
        F = np.zeros([1, n])
        F[0, id[k]] = True
        eibv.append(ExpectedVarianceUsr(Thres, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    row_next, col_next = np.unravel_index(id[ind_desired], (n2, n1))
    return row_next, col_next, id, eibv


rdbu = cm.get_cmap('RdBu', 256)
newcolors = rdbu(np.linspace(0, 1, 35))
newcmp = ListedColormap(newcolors)
EP_prior = EP_1D(mu_prior, Sigma, S_thres)

figpath = "/Users/yaolin/HomeOffice/TAICHI/Square2D/fig/GP/Simulation/EIBV/"
N_steps = 100
row_start, col_start = find_starting_loc_rule(EP_prior)
row_start = 24
col_start = 2
row_now = row_start
col_now = col_start

# row_now = 15
# col_now = 0
row_previous = row_now
col_previous = col_now

mu_posterior = mu_prior
Sigma_posterior = Sigma
noise = tau ** 2
R = np.diagflat(noise)

path_row = []
path_col = []
path_row.append(row_now)
path_col.append(col_now) # only for plotting
accuracy = []

for j in range(N_steps):
    row_cand, col_cand = find_candidates_loc_rule(row_now, col_now)

    # row_next, col_next, id, nibv = find_next_NIBV_rule(row_cand, col_cand, row_now, col_now, row_previous,
    #                                          col_previous, Sigma_posterior, mu_posterior, tau, S_thres)
    row_next, col_next, id, eibv = find_next_EIBV_rule(row_cand, col_cand, row_now, col_now, row_previous,
                                             col_previous, Sigma_posterior, mu_posterior, tau, S_thres)
    # row_next, col_next, id, eibv = find_next_EIBV(row_neighbour, col_neighbour, row_now, col_now,
    #                                     Sigma_posterior, mu_posterior, tau, S_thres)

    row_previous, col_previous = row_now, col_now
    row_now, col_now = row_next, col_next
    path_row.append(row_now)
    path_col.append(col_now)

    ind_next = np.ravel_multi_index((row_next, col_next), (n1, n2))
    F = np.zeros([1, n])
    F[0, ind_next] = True

    y_sampled = np.dot(F, mu_real) + tau * np.random.randn(1).reshape(-1, 1)

    mu_posterior, Sigma_posterior = GPupd(mu_posterior, Sigma_posterior, R, F, y_sampled)
    accuracy.append(np.mean((mu_posterior - mu_real) ** 2))
    ES_Prob = EP_1D(mu_posterior, Sigma_posterior, S_thres)
    ES_Prob_m = np.array(ES_Prob).reshape(n1, n2) # ES_Prob_m is the reshaped matrix form of ES_Prob


    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(nrows=2, ncols=2)

    i = 0
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(ES_Prob_m, vmin = 0, vmax = 1, cmap = newcmp, interpolation="spline36"); plt.title("Excursion probabilities of the updated field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 1
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(mu_posterior.reshape(n1, n2), cmap = newcmp, interpolation="spline36"); plt.title("Realised field")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 2
    axes = fig.add_subplot(gs[i])
    im = axes.imshow(np.sqrt(np.diag(Sigma_posterior)).reshape(n1, n2), cmap = "binary"); plt.title("Uncertainty")
    plt.plot(path_col, path_row, 'r.-', linewidth=2)
    plt.xlabel("s1");plt.ylabel("s2");plt.colorbar(im, fraction=0.045, pad=0.04);

    i = 3
    axes = fig.add_subplot(gs[i])
    plt.title("Mean Squared Error")
    plt.plot(accuracy, 'k.-', linewidth=2)
    plt.xlabel("Iterations")


    plt.savefig(figpath + "E_{:03d}.png".format(j))
    plt.close("all")
    print(j)
    # if not os.path.exists(figpath + "Cond/"):
    #     os.mkdir(figpath + "Cond/")
    # fig.savefig(figpath + "Cond/M_{:03d}.pdf".format(s))



