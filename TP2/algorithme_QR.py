import numpy as np


def algo_QR(input_matrix: np.ndarray):
    n = len(input_matrix[0, :])
    if n == len(input_matrix[:, 0]):
        q = np.zeros((n, n))
        for i in range(n):
            a = np.tile(input_matrix[:, i], (n, 1)).T
            q[:, i] = input_matrix[:, i] - np.multiply(np.tile(np.sum(np.multiply(a, q), axis=0), (n, 1)), q)
            q[:, i] = np.divide(q[:, i], np.sqrt(np.sum(q[:, i])))

    else:
        print("The input needs to be a square matrix")
        quit()
