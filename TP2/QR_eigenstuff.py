import numpy as np
from algorithme_QR import algo_qr, B
from time import time


def QR_eigenstuff(matrix, accuracy):
    n = len(matrix[0, :])
    v = np.eye(n)
    verify_accuracy = np.ones((n, n), dtype=bool) ^ np.eye(n, dtype=bool)
    while np.any(abs(matrix[verify_accuracy]) > accuracy):
        q, r = algo_qr(matrix)
        matrix, v = r @ q, v @ q
    return matrix, v


if __name__ == "__main__":
    error = 10 ** -6
    timer = time()
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    print("L'algorithme avec une erreur de {0} a pris {1} secondes".format(error, time() - timer))
    error = 10 ** -12
    timer = time()
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    print("L'algorithme avec une erreur de {0} a pris {1} secondes".format(error, time() - timer))
    error = 10 ** -18
    timer = time()
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    print("L'algorithme avec une erreur de {0} a pris {1} secondes".format(error, time() - timer))
