import numpy as np
import scipy.constants
from matplotlib import pyplot as plt


def roots(energy):
    return np.sqrt((height - energy) / energy)


def right_hand_side(root, parity):
    if parity == 'even':
        return root
    elif parity == 'odd':
        return -1 / root
    else:
        print('Parity should either be "even" or "odd"')
        quit()


def left_hand_side(energy):
    return np.tan(np.sqrt(omega ** 2 * mass * energy * eV_to_J / (2 * hbar ** 2)))


hbar = scipy.constants.hbar  # [J s]
height = 20  # [eV]
mass = scipy.constants.electron_mass  # [kg]
omega = 1 * 10 ** -9  # [m]
eV_to_J = 1.60218 * 10 ** -19  # [J / eV]

roots = np.vectorize(roots)
RHS = np.vectorize(right_hand_side)
LHS = np.vectorize(left_hand_side)

if __name__ == "__main__":
    energies = np.linspace(0, 20, 50000)  # [eV]
    LHS_graph = LHS(energies)
    LHS_graph[LHS_graph < -10] = np.inf
    LHS_graph[LHS_graph > 10] = np.inf

    plt.figure(1)
    plt.plot(energies, LHS_graph, energies[1:-1], RHS(roots(energies[1:-1]), 'even'), energies[1:-1],
             RHS(roots(energies[1:-1]), 'odd'))
    plt.ylim([-10, 10])
    plt.xlim([0, 20])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid("on", alpha=0.4)
    plt.legend(["y_1", "y_2", "y_3"], loc="upper right", fontsize=16)
    plt.xlabel("Énergie [eV]", fontsize=18)
    plt.ylabel("y [-]", fontsize=18)
    plt.show()
