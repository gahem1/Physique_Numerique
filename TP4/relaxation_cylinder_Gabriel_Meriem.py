import numpy as np


class Cylinder:
    def __init__(self, r_min: float, v_mid: float, r_max: np.ndarray, z_bounds: np.ndarray, step: float):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.previous = None
        num_r = int((r_max.max() - r_min) / step)
        num_z = int((z_bounds[-1] - z_bounds[0]) / step + 2)
        self.grid = np.zeros((num_r, num_z))
        self.unfixed = np.zeros((num_r, num_z), dtype=bool)
        self.rmat = np.tile(np.arange(r_min, r_max, step), (num_z, 1)).T
        for i in range(len(r_max)):
            r = int((r_max[i] - r_min) / step)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step + 1), int((z_bounds[i + 1] - z_bounds[i]) / step + 2)
            self.grid[1:r, z1:z2] = np.tile(np.arange(r, 1, -1) * v_mid / r, (z2 - z1, 1)).T
            self.unfixed[1:r, z1:z2] = np.ones((r - 1, z2 - z1), dtype=bool)

    def iterate(self):
        self.previous, new_grid = np.copy(self.grid), np.copy(self.grid)

        gr1 = (self.grid[1:-1, :-2] + self.grid[1:-1, 2:]) / (2 * (1 + np.square(self.rmat[1:-1, 1:-1])))
        gr2 = self.rmat[2:, 1:-1] * (self.rmat[2:, 1:-1] + self.step / 2) * self.grid[2:, 1:-1] / (
                    2 * (1 + np.square(self.rmat[2:, 1:-1])))
        gr3 = self.rmat[:-2, 1:-1] * (self.rmat[:-2, 1:-1] + self.step / 2) * self.grid[:-2, 1:-1] / (
                    2 * (1 + np.square(self.rmat[:-2, 1:-1])))

        new_grid[1:-1, 1:-1] = gr1 + gr2 + gr3

        self.iteration += 1
        self.grid = np.where(self.unfixed, new_grid, self.previous)


if __name__ == "__main__":
    test = Cylinder(1, 150, np.array([10]), np.array([0, 30]), 0.01)
