import numpy as np


class Perturbation:
    def __init__(self, dim: int, alpha: float, beta: float):
        self.N, self.H, self.W = dim, np.diag(np.arange(0.5, dim + 0.5)), np.zeros((dim, dim))
        self.P, self.coeff = np.copy(self.W), None
        if beta == 0 or beta == alpha ** 2:
            self.coeff = alpha
            fa = np.arange(1, dim - 2)  # smallest factor, used to optimize matrix initialization
            self.W[3:, :-3] += np.diag(np.sqrt(fa * (fa + 1) * (fa + 2)))

            fa = np.arange(dim - 1)
            com = np.sqrt(fa + 1)  # common factor, used to optimize matrix initialization
            self.W[1:, :-1] += np.diag(com * (2 * fa + 2 + com ** 2))

            self.W += self.W.T
            if beta == alpha ** 2:
                fa = np.arange(1, dim - 3)
                self.P[4:, :-4] += np.diag(np.sqrt(fa * (fa + 1) * (fa + 2) * (fa + 3)))

                fa = np.arange(dim - 2)
                com = np.sqrt(fa + 1)
                t = fa * np.sqrt(fa + 2) + (fa + 1) * np.sqrt(fa + 2) + (fa + 2) ** (3 / 2) + (fa + 3) * np.sqrt(fa + 2)
                self.P[2:, :-2] += np.diag(com * t)

                self.P += self.P.T

                fa = np.arange(dim)
                self.P += np.diag(3 * (2 * fa * (fa + 1) + 1))

        elif alpha == 0 or alpha == beta ** 2:
            self.coeff = beta
            fa = np.arange(1, dim - 3)
            self.W[4:, :-4] += np.diag(np.sqrt(fa * (fa + 1) * (fa + 2) * (fa + 3)))

            fa = np.arange(dim - 2)
            com = np.sqrt(fa + 1)
            ter = fa * np.sqrt(fa + 2) + (fa + 1) * np.sqrt(fa + 2) + (fa + 2) ** (3 / 2) + (fa + 3) * np.sqrt(fa + 2)
            self.W[2:, :-2] += np.diag(com * ter)

            self.W += self.W.T

            fa = np.arange(dim)
            self.W += np.diag(3 * (2 * fa * (fa + 1) + 1))
            if alpha == beta ** 2:
                fa = np.arange(1, dim - 2)
                self.P[3:, :-3] += np.diag(np.sqrt(fa * (fa + 1) * (fa + 2)))

                fa = np.arange(dim - 1)
                com = np.sqrt(fa + 1)
                self.P[1:, :-1] += np.diag(com * (2 * fa + 2 + com ** 2))

                self.P += self.P.T

        else:
            self.coeff = alpha
            multiple = beta / alpha  # Coefficient by which X^4 is multiplied
            fa = np.arange(1, dim - 3)
            self.W[4:, :-4] += np.diag(multiple * np.sqrt(fa * (fa + 1) * (fa + 2) * (fa + 3)))

            fa = np.arange(1, dim - 2)
            self.W[3:, :-3] += np.diag(np.sqrt(fa * (fa + 1) * (fa + 2)))

            fa = np.arange(dim - 2)
            com = np.sqrt(fa + 1)
            ter = fa * np.sqrt(fa + 2) + (fa + 1) * np.sqrt(fa + 2) + (fa + 2) ** (3 / 2) + (fa + 3) * np.sqrt(fa + 2)
            self.W[2:, :-2] += np.diag(multiple * com * ter)

            fa = np.arange(dim - 1)
            com = np.sqrt(fa + 1)
            self.W[1:, :-1] += np.diag(com * (2 * fa + 2 + com ** 2))

            self.W += self.W.T

            fa = np.arange(dim)
            self.W += np.diag(3 * multiple * (2 * fa * (fa + 1) + 1))

    def first_order_energy(self):
        return self.coeff * np.diag(self.W)

    def first_order_state(self):
        states = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    states[i, j] = self.W[i, j] / (j - i)
        return self.coeff * states

    def second_order_energy(self):
        energies = np.zeros(self.N)
        for j in range(self.N):
            for i in range(self.N):
                if i != j:
                    energies[j] += self.W[i, j] ** 2 / (j - i)
        energies += np.diag(self.P)
        return self.coeff ** 2 * energies

    def second_order_state(self, fos=None):
        states = np.zeros((self.N, self.N))
        if fos is None:
            fos = self.first_order_state()
        vector_elements = self.W @ fos
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    states[i, j] = self.P[i, j] / (j - i)
                    states[i, j] += vector_elements

        return self.coeff ** 2 * states
