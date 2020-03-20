from oscillateur_anharmonique_Gabriel_Meriem import *


if __name__ == "__main__":
    omega = 1
    xi = 1
    xpi = 0

    xharm1, xpharm1 = rk4(N, xi, xpi, omega, xpp)
    xharm2, xpharm2 = rk4(N, xi * 2, xpi, omega, xpp)
    xanharm1, xpanharm1 = rk4(N, xi * 0.5, xpi, omega, xpp)
    xanharm2, xpanharm2 = rk4(N, xi * 3, xpi, omega, xpp)

    plt.figure()
    plt.plot(xharm1, xpharm1, xharm2, xpharm2, xanharm1, xpanharm1, xanharm2, xpanharm2)
    plt.ylabel("xp", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(["Harmonique avec x(0)={}".format(xi), "Harmonique avec x(0)={}".format(2 * xi),
                "Anharmonique avec x(0)={}".format(0.5 * xi), "Anharmonique avec x(0)={}".format(3 * xi)],
               loc="upper right", fontsize=8)
    plt.show()
