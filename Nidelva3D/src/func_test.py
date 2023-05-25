
# GRF =======================================
sigma = 1.0
nugget = .4
eta = 4.5 / .7
threshold = 27.8

N = 25
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

grid = np.stack((X.flatten(), Y.flatten()), axis=1)
Ngrid = grid.shape[0]

dm = cdist(grid, grid, metric='euclidean')
Sigma = sigma ** 2 * (1 + eta * dm) * np.exp(-eta * dm)

mu = np.linalg.cholesky(Sigma) @ np.random.randn(Sigma.shape[0]).reshape(-1, 1)
mu += np.ones_like(mu) * 28

import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap

plt.scatter(grid[:, 0], grid[:, 1], c=mu, cmap=get_cmap("BrBG", 10))
plt.colorbar()
plt.show()

ind_measured = 10
F = np.zeros([1, Ngrid])
F[0, ind_measured] = True
R = np.eye(1) * nugget
C = F @ Sigma @ F.T + R
VR = Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
Sigma_posterior = Sigma - VR

sigma_diag = np.diag(Sigma_posterior)
vr_diag = np.diag(VR)
mu_input = mu.squeeze()
sigma_diag_input = sigma_diag
vr_diag_input = vr_diag
# ============================================


plt.figure()
plt.subplot(311)
plt.plot(EBV1, label="Analytical")
plt.subplot(312)
plt.plot(EBV2, label="Fast")
plt.subplot(313)
plt.plot(np.array(EBV1) - np.array(EBV2).squeeze(), label="Difference")
plt.axhline(y=np.mean(np.array(EBV1) - np.array(EBV2)), color='r', linestyle='--')
print("y=", np.mean(np.array(EBV1) - np.array(EBV2)))
plt.legend()
plt.show()

print("eibv_temp1: ", eibv_temp1)
print("eibv_temp2: ", eibv_temp2)