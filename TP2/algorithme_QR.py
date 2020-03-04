import numpy as np


def algo_qr(input_matrix: np.ndarray):
    n = len(input_matrix[0, :])
    if n == len(input_matrix[:, 0]):
        q, r = np.zeros((n, n)), np.zeros((n, n))
        u = input_matrix[:, 0]
        r[0, 0] = np.sqrt(np.sum(u * u))
        q[:, 0] = u / r[0, 0]
        for i in range(1, n):
            a = np.tile(input_matrix[:, i], (i, 1)).T
            r[:i, i] = np.sum(a * q[:, :i], axis=0)
            u = input_matrix[:, i] - np.sum(np.tile(r[:i, i], (n, 1)) * q[:, :i], axis=1)
            r[i, i] = np.sqrt(np.sum(u * u))
            q[:, i] = u / r[i, i]
        return q, r
    else:
        print("The input needs to be a square matrix")
        quit()


if __name__ == "__main__":
    B = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9], [4, 7, 9, 2]])
    Q, R = algo_qr(B)
    print(Q @ R)
