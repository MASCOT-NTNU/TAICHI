import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from pykdtree.kdtree import KDTree as KDTree_pykdtree
import timeit
import matplotlib.pyplot as plt
import time

n = 1000
xv = np.linspace(0, 1, n)
yv = np.linspace(0, 1, n)
zv = np.linspace(0, 1, n)
xx, yy, zz = np.meshgrid(xv, yv, zv)
xx = xx.flatten()
yy = yy.flatten()
zz = zz.flatten()
grid = np.vstack((xx, yy, zz)).T

query_points = np.random.rand(100000, 3)

tree = KDTree(grid)
tree_pykdtree = KDTree_pykdtree(grid)
def get_ind_from_locations(loc: np.ndarray) -> np.ndarray:
    """
    Get the indices of the closest grid point to the given locations.
    """
    # Get the distances and indices of the closest grid point to the given locations
    dist = cdist(loc, grid)
    ind = np.argmin(dist, axis=1)
    return ind

# Method 1: KDTree from scipy
t1 = time.time()
dist1, ind1 = tree.query(query_points, k=1)
t2 = time.time()
print(f"Time used for KDTree from scipy: {t2-t1}")

# Method 2: KDTree from pykdtree
t1 = time.time()
dist2, ind2 = tree_pykdtree.query(query_points, k=1)
t2 = time.time()
print(f"Time used for KDTree from pykdtree: {t2-t1}")

# Method 3: get distance and indices from cdist
t1 = time.time()
ind3 = get_ind_from_locations(query_points)
t2 = time.time()
print(f"Time used for cdist: {t2-t1}")

# Method 4: KDTree from pykdtree with multi-threading
# t1 = time.time()
# for i in range(100):
#     dist4, ind4 = tree_pykdtree.query(query_points)
# t2 = time.time()
# print(f"Time used for KDTree from pykdtree with multi-threading: {t2-t1}")

# Check if the indices are the same
np.all(ind1 == ind2)
np.all(ind1 == ind3)
np.all(ind2 == ind3)


