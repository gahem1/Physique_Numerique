import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Cylinder:
    def __init__(self, r_min, v_mid, r_max, z_bounds, step, error, ztige):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.previous = None
        self.error = error + 1
        num_r = int(r_max.max() * 2 / step + 3)
        num_z = int((z_bounds[-1] - z_bounds[0]) / step + 3)
        self.grid = np.zeros((num_r, num_z))
        self.unfixed = np.zeros((num_r, num_z), dtype=bool)
        self.rmat = np.tile(np.arange(step / 2, r_max.max(), step / 2), (num_z - 2, 1)).T
        self.rmat = self.step / self.rmat
        r1 = int(r_min * 2 / step + 3)
        for i in range(len(r_max)):
            r2 = int((r_max[i]) * 2 / step + 2)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step + 1), int((z_bounds[i + 1] - z_bounds[0]) / step + 2)
            self.grid[r1:r2, z1:z2] = np.tile(np.arange(r2 - r1, 0, -1) * v_mid / (r2 - r1), (z2 - z1, 1)).T
            self.grid[r2 - 1:, z1:z2] = 0
            self.unfixed[:r2 - 2, z1:z2] = bool(1)

        z1, z2 = int((ztige[0] - z_bounds[0]) / step), int((ztige[-1] - z_bounds[0]) / step + 2)
        self.grid[0:r1, z1:z2] = v_mid
        self.unfixed[:r1, z1:z2] = bool(0)

    def iterate(self):
        self.previous, new = np.copy(self.grid), np.copy(self.grid)
        gr1 = self.grid[4:, 1:-1] + self.grid[:-4, 1:-1] + self.grid[2:-2, 2:] + self.grid[2:-2, :-2]
        gr2 = (self.grid[3:-1, 1:-1] - self.grid[1:-3, 1:-1]) * self.rmat
        new[2:-2, 1:-1] = (gr1 + gr2) / 4

        self.iteration += 1
        self.grid = np.where(self.unfixed, new, self.previous)
        self.error = np.absolute(self.grid - self.previous).max()


if __name__ == "__main__":
    err = 0.01
    h = 2 * err
    cyl = Cylinder(1, 150, np.array([10, 8, 4]), np.array([0, 10, 22, 33]), h, err, np.array([2, 32]))
    while cyl.error >= err:
        cyl.iterate()

    print(cyl.iteration)
    ax = sns.heatmap(cyl.grid[-1:1:-1, 1:-1], cbar_kws={'label': 'Voltage [V]'})
    ax.set_xlabel("z [cm]", fontsize=20)
    ax.set_ylabel("r [cm]", fontsize=20)
    ax.set_xticks(np.arange(0.5, 1750.5, 250))
    ax.set_xticklabels(np.arange(0, 35, 5), fontsize=18)
    ax.set_yticks(np.arange(0, 1100, 100))
    ax.set_yticklabels(np.arange(10, -1, -1), fontsize=18)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    plt.show()
