import numpy as np
from time import time


class Operator:
    def __init__(self, dim: int, alpha: float, beta: float, prev_matrix: np.ndarray = None):
        self.N, self.vep = dim, np.eye(dim)
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
        for j in range(self.N):
            val = r[j, j]
            for i in range(self.N - 1, j, -1):
                if r[i, j] != 0:
                    rot, norm = np.eye(self.N), np.sqrt(val * val + r[i, j] * r[i, j])
                    rot[j, j] = val / norm
                    rot[i, self.N - 1] = rot[0, j]
                    rot[i, j] = r[i, j] / norm
                    rot[j, self.N - 1] = -rot[i, j]
                    q, r = rot @ q, rot @ r
        return q.T, r

    def rayleigh_iteration(self, vap, vep):
        nvap, nsing = vap, True
        for i in range(self.N):
            nmat = self.matrix - nvap[i] * np.eye(self.N)
            if np.linalg.det(nmat) != 0:
                vep[:, i] = np.linalg.inv(nmat) @ vep[:, i]
                vep[:, i] = vep[:, i] / np.sqrt(np.sum(vep[:, i] * vep[:, i]))
            else:
                nsing = False
                break

            nvap = np.sum(vep * (self.matrix @ vep), axis=0)

        diff = np.max(abs(nvap - vap))
        return nvap, vep, diff, nsing

    def eigenalgo(self, version: str = "Rayleigh", accuracy: float= 0, cap: int= 50000):
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

        elif version == "Givens":  # Warning: Givens rotations algorithm does not appear to work properly
            temps = time()
            while np.any(abs(self.vap[verify_accuracy]) > accuracy) and j < cap:
                j += 1
                q, r = self.givens_qr()
                self.vap, self.vep = r @ q, self.vep @ q

        elif version == "Rayleigh":
            vap_guess, vep_guess, not_sing, diff = np.arange(0.5, self.N + 0.5), np.eye(self.N), True, accuracy + 1
            temps = time()
            while diff > accuracy and j < cap and not_sing:
                j += 1
                vap_guess, vep_guess, diff, not_sing = self.rayleigh_iteration(vap_guess, vep_guess)

            temps = time() - temps
            if not_sing:
                return vap_guess, vep_guess, diff, j, temps
            else:
                return vap_guess, vep_guess, 0, j - 1, temps

        else:
            print("Please select an appropriate value for the version parameter")

        temps = time() - temps
        diff = np.max(abs(self.vap[verify_accuracy]))
        return np.diag(self.vap), self.vep, diff, j, temps
