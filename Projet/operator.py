from numpy import *


class Operator:
    def __init__(self, dim: int, alpha: float, beta: float, num: int):
        self.N = dim
        self.matrix = zeros((dim, dim))

    def algo_qr(self):
        q, r = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        u = copy(self.matrix[:, 0])
        r[0, 0] = sqrt(sum(u * u))
        q[:, 0] = u / r[0, 0]
        for i in range(1, self.N):
            a = tile(self.matrix[:, i], (i, 1)).T
            r[:i, i] = sum(a * q[:, :i], axis=0)
            u = self.matrix[:, i] - sum(tile(r[:i, i], (self.N, 1)) * q[:, :i], axis=1)
            r[i, i] = sqrt(sum(u * u))
            q[:, i] = u / r[i, i]
        return q, r
