import numpy as np


class Operator:
    def __init__(self, dim: int, alpha: float, beta: float):
        self.N = dim
        self.matrix = np.zeros((dim, dim))

        fac = np.arange(1, dim - 3)
        self.matrix[4:, :-4] += np.diag(beta * np.sqrt(fac * (fac + 1) * (fac + 2) * (fac + 3)))

        fac = np.arange(1, dim - 2)
        self.matrix[3:, :-3] += np.diag(alpha * np.sqrt(fac * (fac + 1) * (fac + 2)))

        fac = np.arange(dim - 1)
        ptwo +=

        pone +=

        n +=

        mone +=

        mtwo +=

        mthree +=

        mfour +=


    def algo_qr(self):
        q, r = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        u = np.copy(self.matrix[:, 0])
        r[0, 0] = np.sqrt(sum(u * u))
        q[:, 0] = u / r[0, 0]
        for i in range(1, self.N):
            a = np.tile(self.matrix[:, i], (i, 1)).T
            r[:i, i] = np.sum(a * q[:, :i], axis=0)
            u = self.matrix[:, i] - np.sum(np.tile(r[:i, i], (self.N, 1)) * q[:, :i], axis=1)
            r[i, i] = np.sqrt(np.sum(u * u))
            q[:, i] = u / r[i, i]
        return q, r
