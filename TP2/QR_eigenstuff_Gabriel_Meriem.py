import numpy as np
from algorithme_QR_Gabriel_Meriem import algo_qr, B
from time import time


def QR_eigenstuff(matrix, accuracy):
    n = len(matrix[0, :])
    v = np.eye(n)
    if accuracy <= 10 ** -16:
        matrix = matrix.astype(np.longdouble)
        v = matrix.astype(np.longdouble)

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
    print("L'algorithme avec une erreur de {0} a pris {1:f} secondes".format(error, time() - timer))
    error = 10 ** -12
    timer = time()
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    print("L'algorithme avec une erreur de {0} a pris {1:f} secondes".format(error, time() - timer))
    error = 10 ** -18
    print(error)
    timer = time()
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    print("L'algorithme avec une erreur de {0} a pris {1:f} secondes".format(error, time() - timer))
