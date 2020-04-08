from relaxation_cylinder_Gabriel_Meriem import *
from gauss_seidel_Gabriel_Meriem import Gauss
from over_relaxation_Gabriel_Meriem import Failure
from over_gauss_Gabriel_Meriem import OverGauss

if __name__ == "__main__":
    x = np.arange(1, 601)
    err = np.empty((4, 600))
    tim = np.empty((4, 600))
    it = np.empty(4)

    error = 10 ** -8
    h = 0.05

    it[0] = Cylinder(1, 150, np.array([10]), np.array([0, 30]), h, error, np.array([0, 30]))
    it[1] = Gauss(1, 150, np.array([10]), np.array([0, 30]), h, error, np.array([0, 30]))
    it[2] = OverGauss(1, 150, np.array([10]), np.array([0, 30]), h, error, np.array([0, 30]), 0.9435)
    it[3] = Failure(1, 150, np.array([10]), np.array([0, 30]), h, error, np.array([0, 30]), 0.001)
    for i in range(4):
        debut = time()
        for j in range(600):
            it[i].iterate()
            err[i, j] = it[i].error
            tim[i, j] = time() - debut

    plt.plot(x, err[0, :], 'r', x, err[1, :], 'g', x, err[2, :], 'b', x, err[3, :], 'y')
    plt.xlabel("Nombre d'itérations", fontsize=18)
    plt.ylabel("Erreur maximale [V]", fontsize=18)

    plt.plot(x, tim[0, :], 'r', x, tim[1, :], 'g', x, tim[2, :], 'b', x, tim[3, :], 'y')
    plt.xlabel("Nombre d'itérations", fontsize=18)
    plt.ylabel("Erreur maximale [V]", fontsize=18)
