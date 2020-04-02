import numpy as np


class Cylinder:
    def __init__(self, r_min: float, v_mid: float, r_max: np.ndarray, z_bounds: np.ndarray, step: float, error: float):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.previous = None
        self.error = error + 1
        num_r = int((r_max.max() - r_min) / step + 1)
        num_z = int((z_bounds[-1] - z_bounds[0]) / step + 3)
        self.grid = np.zeros((num_r, num_z))
        self.unfixed = np.zeros((num_r, num_z), dtype=bool)
        self.rmat = np.tile(np.arange(r_min, r_max.max() + step, step), (num_z, 1)).T
        self.rmats = np.square(self.rmat)
        self.rmat, self.rmats = self.rmat[1:-1, 1:-1], self.rmats[1:-1, 1:-1]
        for i in range(len(r_max)):
            r = int((r_max[i] - r_min) / step + 1)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step + 1), int((z_bounds[i + 1] - z_bounds[0]) / step + 2)
            self.grid[0:r, z1:z2] = np.tile(np.arange(r, 0, -1) * v_mid / r, (z2 - z1, 1)).T
            self.unfixed[1:r - 1, z1:z2] = np.ones((r - 2, z2 - z1), dtype=bool)

    def iterate(self):
        self.previous, new_grid = np.copy(self.grid), np.copy(self.grid)

        gr1 = (self.grid[1:-1, :-2] + self.grid[1:-1, 2:]) / (2 * (1 + self.rmats))
        gr2 = self.rmat * (self.rmat + self.step / 2) * self.grid[2:, 1:-1] / (2 * (1 + self.rmats))
        gr3 = self.rmat * (self.rmat + self.step / 2) * self.grid[:-2, 1:-1] / (2 * (1 + self.rmats))

        new_grid[1:-1, 1:-1] = gr1 + gr2 + gr3

        self.iteration += 1
        self.grid = np.where(self.unfixed, new_grid, self.previous)
        self.error = np.absolute(self.grid - self.previous).max()


if __name__ == "__main__":
    test = Cylinder(1, 150, np.array([10]), np.array([0, 30]), 0.01, 2 * 10 ** -8)
