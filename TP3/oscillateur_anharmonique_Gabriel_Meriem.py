from oscillateur_harmonique_Gabriel_Meriem import *


def newxpp(x1, omega):
    return -x1 ** 3 * omega ** 2


if __name__ == "__main__":
    xi = 1
    xpi = 0
    freq = 1

    plt.figure()
    plt.plot(tvals, rk4(N, xi, xpi, freq, newxpp)[0], tvals, rk4(N, 1.01 * xi, xpi, freq, newxpp)[0], tvals,
             rk4(N, 1.02 * xi, xpi, freq, newxpp)[0], linewidth=2)
    plt.ylabel("x(t)", fontsize=18)
    plt.xlabel("t", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["x(0)={}".format(xi), "x(0)={}".format(1.01 * xi), "x(0)={}".format(1.02 * xi)], fontsize=12,
               loc='upper right')
    plt.show()
