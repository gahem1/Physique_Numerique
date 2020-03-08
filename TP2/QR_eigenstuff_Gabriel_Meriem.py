import numpy as np
from algorithme_QR_Gabriel_Meriem import algo_qr, B


def QR_eigenstuff(matrix, accuracy):
    n = len(matrix[0, :])
    v = np.eye(n)
    cap = 10000

    verify_accuracy = np.ones((n, n), dtype=bool) ^ np.eye(n, dtype=bool)
    j = 0
    while np.any(abs(matrix[verify_accuracy]) > accuracy) and j < cap:
        j += 1
        q, r = algo_qr(matrix)
        matrix, v = r @ q, v @ q

    if j == cap:
        print("Attention: l'erreur sera plus grande qu'attendu")

    return matrix, v


if __name__ == "__main__":
    error = 10 ** -6
    print("Pour une erreur de {}".format(error))
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    error = 10 ** -12
    print("Pour une erreur de {}".format(error))
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
    error = 10 ** -18
    print("Pour une erreur de {}".format(error))
    vep, vap = QR_eigenstuff(B, error)
    print(vep)
    print(vap)
