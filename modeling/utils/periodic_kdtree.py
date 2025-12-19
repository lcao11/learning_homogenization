import numpy as np
from scipy.spatial import cKDTree


class PeriodicCKDTree():
    def __init__(self, points):
        dim = points.shape[1]
        assert dim == 2 or dim == 3
        self._n_points = points.shape[0]
        shifts = [-1, 0, 1]
        extended_points = []
        if dim == 2:
            for shift_1 in shifts:
                for shift_2 in shifts:
                    new_points = points.copy()
                    new_points[:, 0] += shift_1
                    new_points[:, 1] += shift_2
                    extended_points.append(new_points)
        if dim == 3:
            for shift_1 in shifts:
                for shift_2 in shifts:
                    for shift_3 in shifts:
                        new_points = points.copy()
                        new_points[:, 0] += shift_1
                        new_points[:, 1] += shift_2
                        new_points[:, 2] += shift_3
                        extended_points.append(new_points)
        self._kdtree = cKDTree(np.concatenate(extended_points, axis=0))

    def query(self, x):
        distance, id = self._kdtree.query(x)
        return distance, id % self._n_points
