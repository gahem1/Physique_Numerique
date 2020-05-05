import numpy as np
from time import time


class Operator:
    def __init__(self, dim: int, alpha: float, beta: float, prev_matrix: np.ndarray = None):
        self.N, self.vep, self.calc = dim, np.eye(dim), np.ones(dim, dtype=bool)
        if prev_matrix is None:
            self.matrix = np.zeros((dim, dim))

            fa = np.arange(1, dim - 3)  # smallest factor, used to optimize matrix initialization
            self.matrix[4:, :-4] += np.diag(beta * np.sqrt(fa * (fa + 1) * (fa + 2) * (fa + 3)))

            fa = np.arange(1, dim - 2)
            self.matrix[3:, :-3] += np.diag(alpha * np.sqrt(fa * (fa + 1) * (fa + 2)))

            fa = np.arange(dim - 2)
            com = np.sqrt(fa + 1)  # common factor, used to optimize matrix initialization
            ter = fa * np.sqrt(fa + 2) + (fa + 1) * np.sqrt(fa + 2) + (fa + 2) ** (3/2) + (fa + 3) * np.sqrt(fa + 2)
            self.matrix[2:, :-2] += np.diag(beta * com * ter)

            fa = np.arange(dim - 1)
            com = np.sqrt(fa + 1)
            self.matrix[1:, :-1] += np.diag(alpha * com * (2 * fa + 2 + com ** 2))

            self.matrix += self.matrix.T

            fa = np.arange(dim)
            self.matrix += np.diag(3 * beta * (2 * fa * (fa + 1) + 1) + fa + 0.5)

        else:
            self.matrix = prev_matrix[:dim, :dim]

        self.vap = np.copy(self.matrix)

    def gram_schmidt_qr(self):
        q, r = np.zeros((self.N, self.N)), np.zeros((self.N, self.N))
        u = np.copy(self.vap[:, 0])
        r[0, 0] = np.sqrt(sum(u * u))
        q[:, 0] = u / r[0, 0]
        for i in range(1, self.N):
            a = np.tile(self.vap[:, i], (i, 1)).T
            r[:i, i] = np.sum(a * q[:, :i], axis=0)
            u = self.vap[:, i] - np.sum(np.tile(r[:i, i], (self.N, 1)) * q[:, :i], axis=1)
            r[i, i] = np.sqrt(np.sum(u * u))
            q[:, i] = u / r[i, i]

        return q, r

    def givens_qr(self):
        q, r = np.eye(self.N), np.copy(self.vap)
        for j in range(self.N - 1):
            for i in range(self.N - 1, j, -1):
                if self.vap[i, j] != 0:
                    val, rot = r[j, j], np.eye(self.N)
                    norm = np.sqrt(val * val + r[i, j] * r[i, j])
                    rot[j, j] = val / norm
                    rot[i, i] = rot[j, j]
                    rot[j, i] = r[i, j] / norm
                    rot[i, j] = -rot[j, i]
                    q, r = rot @ q, rot @ r

        return q.T, r

    def rayleigh_iteration(self, vap, vep):
        nsing, diff = True, 0
        for i in np.arange(self.N)[self.calc]:
            nmat = self.matrix - vap[i] * np.eye(self.N)
            if np.linalg.det(nmat) != 0:
                vep[:, i] = np.linalg.inv(nmat) @ vep[:, i]
                vep[:, i] = vep[:, i] / np.sqrt(np.sum(vep[:, i] * vep[:, i]))
            else:
                self.calc[i] -= 1

        nvap = np.sum(vep * (self.matrix @ vep), axis=0)
        diff = np.sum(np.abs(nvap - vap))
        if np.sum(self.calc) == 0:
            nsing, diff = False, 0

        return nvap, vep, diff, nsing

    def reset_vap_vep(self):
        self.vap, self.vep = np.copy(self.matrix), np.eye(self.N)

    def eigenalgo(self, accuracy: float = 0, cap: int = 50000, version: str = "Givens"):
        """
        Uses the desired algorithm to find eigenvalues and eigenvectors of Operator object's matrix
        Returns eigenvalues, eigenvectors, error, number of iterations, and duration in this order
        """
        j, temps, verify_accuracy = 0, 0, np.ones((self.N, self.N), dtype=bool) ^ np.eye(self.N, dtype=bool)
        if version == "Gram-Schmidt":
            temps = time()
            while np.any(abs(self.vap[verify_accuracy]) > accuracy) and j < cap:
                j += 1
                q, r = self.gram_schmidt_qr()
                self.vap, self.vep = r @ q, self.vep @ q

        elif version == "Givens":
            verify_accuracy = np.ones((self.N, self.N), dtype=bool) ^ np.eye(self.N, dtype=bool)
            temps = time()
            while np.any(abs(self.vap[verify_accuracy]) > accuracy) and j < cap:
                j += 1
                q, r = self.givens_qr()
                self.vap, self.vep = r @ q, self.vep @ q

        elif version == "Rayleigh":
            vap_guess, vep_guess, not_sing, diff = np.arange(0.5, self.N + 0.5), np.eye(self.N), True, accuracy + 1
            cond, j, memorize, temps = True, 0, np.zeros(self.N), time()
            while cond:  # Stop condition, all eigenvalues must be different
                while diff > accuracy and j < cap and not_sing:
                    j += 1
                    vap_guess, vep_guess, diff, not_sing = self.rayleigh_iteration(vap_guess, vep_guess)

                self.calc, cond, first, not_sing = np.zeros(self.N, dtype=bool), False, True, True
                for i in range(self.N):
                    if np.sum(np.less(np.abs(vap_guess - vap_guess[i]), 10 ** -6)) != 1:
                        vap_guess[i + 1:] += 1 + memorize[i]
                        if first:
                            memorize[i] += 1
                            vep_guess[i + 1:, i + 1:] = np.eye(self.N - i - 1)
                            first, cond, diff = False, True, accuracy + 1
                            self.calc[i + 1:] = 1

            temps = time() - temps
            return vap_guess, vep_guess, diff, j, temps

        else:
            print("Please select an appropriate value for the version parameter")

        temps = time() - temps
        diff = np.max(abs(self.vap[verify_accuracy]))
        return np.diag(self.vap), self.vep, diff, j, temps


if __name__ == "__main__":
    test = Operator(10, 0.01, 0.01)
    va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Gram-Schmidt")
    print("Gram-Schmidt")
    print(va)
    print("Diff = {}".format(di))
    print("j = {}".format(gg))
    print("Temps = {}".format(te))
    test = Operator(10, 0.01, 0.01)
    va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Givens")
    print("Givens")
    print(va)
    print("Diff = {}".format(di))
    print("j = {}".format(gg))
    print("Temps = {}".format(te))
    test = Operator(10, 0.01, 0.01)
    va, ve, di, gg, te = test.eigenalgo(10 ** -14, 10000, "Rayleigh")
    print("Rayleigh")
    print(va)
    print("Diff = {}".format(di))
    print("j = {}".format(gg))
    print("Temps = {}".format(te))
