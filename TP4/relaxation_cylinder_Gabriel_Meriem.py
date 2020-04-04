import numpy as np


class Cylinder:
    def __init__(self, r_min: float, v_mid: float, r_max: np.ndarray, z_bounds: np.ndarray, step: float, error: float):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.previous = None
        self.error = error + 1
        num_r = int((r_max.max() - r_min) * 2 / step + 3)
        num_z = int((z_bounds[-1] - z_bounds[0]) / step + 3)
        self.grid = np.zeros((num_r, num_z))
        self.unfixed = np.zeros((num_r, num_z), dtype=bool)
        self.rmat = np.tile(np.arange(r_min - step / 2, r_max.max() + step, step / 2), (num_z, 1)).T
        self.rmat = self.step / self.rmat[2:-2, 1:-1]
        for i in range(len(r_max)):
            r = int((r_max[i] - r_min) * 2 / step + 2)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step + 1), int((z_bounds[i + 1] - z_bounds[0]) / step + 2)
            self.grid[1:r, z1:z2] = np.tile(np.arange(r - 1, -1, -1) * v_mid / r, (z2 - z1, 1)).T
            self.grid[r:, z1:z2] = 0
            self.unfixed[2:r - 2, z1:z2] = np.ones((r - 4, z2 - z1), dtype=bool)
        self.grid[0:2, :] = v_mid

    def iterate(self):
        self.previous, new_grid = np.copy(self.grid), np.copy(self.grid)

        gr1 = self.grid[4:, 1:-1] + self.grid[:-4, 1:-1] + self.grid[2:-2, 2:], self.grid[2:-2, :-2]
        gr2 = (self.grid[3:-1, 1:-1] - self.grid[1:-3, 1:-1]) * self.rmat
        new_grid[1:-1, 1:-1] = (gr1 + gr2) / 4

        self.iteration += 1
        self.grid = np.where(self.unfixed, new_grid, self.previous)
        self.error = np.absolute(self.grid - self.previous).max()


if __name__ == "__main__":
    err = 2 * 10 ** -8
    test = Cylinder(1, 150, np.array([10]), np.array([0, 30]), 1.25 * 10 ** -5, err)
    while test.error >= err:
        test.iterate()

    print(test.error)
    print(test.iteration)
