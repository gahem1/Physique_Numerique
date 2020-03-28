import numpy as np


class Cylinder:
    def __init__(self, r_min: float, v_mid: float, r_max: np.ndarray, z_bounds: np.ndarray, step: float):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.num_r = int((r_max.max() - r_min) / step)
        self.num_z = int((z_bounds[-1] - z_bounds[1]) / step)
        self.previous = None
        self.grid = np.zeros((self.num_r, self.num_z))
        self.unfixed = np.zeros((self.num_r, self.num_z), dtype=bool)
        self.rmat = np.tile(np.arange(r_min, r_max, step), self.num_z).T
        for i in range(len(r_max)):
            r = int((r_max[i] - r_min) / step)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step), int((z_bounds[i + 1] - z_bounds[i]) / step)
            self.grid[1:r, z1:z2 + 1] = np.tile(np.arange(r - 1, 1, step) * v_mid / r, (z2 - z1 + 1, 1)).T
            self.unfixed[1:r, z1:z2 + 1] = np.ones((r - 1, z2 - z1 + 1), dtype=bool)

    def iterate(self):
        self.previous, new_grid = np.copy(self.grid), np.copy(self.grid)
        new_grid[1:-1, 1:-1] = (self.grid[:-2, 1:-1] + self.grid[2:, 1:-1] + self.grid[1:-1, :-2] +
                                self.grid[1:-1, 2:]) / (2 * (1 + np.square(self.rmat)))
        self.iteration += 1
        self.grid = np.where(self.unfixed, new_grid, self.previous)
