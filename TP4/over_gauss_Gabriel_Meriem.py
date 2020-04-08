import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time


class OverGauss:
    def __init__(self, r_min, v_mid, r_max, z_bounds, step, error, ztige, omega):
        self.iteration = 0
        self.v_mid = v_mid
        self.step = step
        self.error = error + 1
        self.cible = error
        self.omega = omega
        self.num_r = int(r_max.max() * 2 / step + 3)
        self.num_z = int((z_bounds[-1] - z_bounds[0]) / step + 3)
        self.grid = np.zeros((self.num_r, self.num_z))
        self.unfixed = np.zeros((self.num_r, self.num_z), dtype=bool)
        self.rmat = np.tile(np.arange(step / 2, r_max.max(), step / 2), (self.num_z - 2, 1)).T
        self.rmat = self.step / self.rmat
        r1 = int(r_min * 2 / step + 3)
        for i in range(len(r_max)):
            r2 = int((r_max[i]) * 2 / step + 2)
            z1, z2 = int((z_bounds[i] - z_bounds[0]) / step + 1), int((z_bounds[i + 1] - z_bounds[0]) / step + 2)
            self.grid[r1:r2, z1:z2] = np.tile(np.arange(r2 - r1, 0, -1) * v_mid / (r2 - r1), (z2 - z1, 1)).T
            self.grid[r2 - 1:, z1:z2] = 0
            self.unfixed[:r2 - 2, z1:z2] = bool(1)

        z1, z2 = int((ztige[0] - z_bounds[0]) / step), int((ztige[-1] - z_bounds[0]) / step + 2)
        self.grid[:r1, z1:z2] = v_mid
        self.unfixed[:r1, z1:z2] = bool(0)

    def iterate(self):
        self.error = self.cible
        for i in range(3, self.num_r - 1):
            for j in range(1, self.num_z):
                if self.unfixed[i, j]:
                    add = self.grid[i + 2, j] + self.grid[i - 2, j] + self.grid[i, j - 1] + self.grid[i, j + 1]
                    add += (self.grid[i + 1, j] - self.grid[i - 1, j]) * self.rmat[i - 2, j - 1]
                    add /= 4
                    if abs(add - self.grid[i, j]) > self.error:
                        self.error = abs(add - self.grid[i, j])

                    self.grid[i, j] = (1 + self.omega) * add - self.omega * self.grid[i, j]

        self.iteration += 1


if __name__ == "__main__":
    err = 0.05
    h = 2 * err

    cyl = OverGauss(1, 150, np.array([10]), np.array([0, 30]), h, err, np.array([0, 30]), 0.9435)
    debut = time()
    while cyl.error > err:
        cyl.iterate()

    print(time() - debut)
    print(cyl.iteration)

    ax = sns.heatmap(cyl.grid[-1:1:-1, 1:-1], cbar_kws={'label': 'Voltage [V]'})
    ax.set_xlabel("z [cm]", fontsize=20)
    ax.set_ylabel("r [cm]", fontsize=20)
    ax.set_xticks(np.arange(0.5, 350.5, 50))
    ax.set_xticklabels(np.arange(0, 35, 5), fontsize=18)
    ax.set_yticks(np.arange(0, 200, 20))
    ax.set_yticklabels(np.arange(10, 0, -1), fontsize=18)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    plt.show()
