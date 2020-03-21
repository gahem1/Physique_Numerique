from oscillateur_anharmonique_Gabriel_Meriem import *


if __name__ == "__main__":
    omega = 1
    xi = 1
    xpi = 0
    diff = 1.5

    xharm1, xpharm1 = rk4(N, xi, xpi, omega, xpp)
    xanharm1, xpanharm1 = rk4(N, xi, xpi, omega, newxpp)
    xharm2, xpharm2 = rk4(N, xi * diff, xpi, omega, xpp)
    xanharm2, xpanharm2 = rk4(N, xi * diff, xpi, omega, newxpp)

    plt.figure()
    plt.plot(xharm1, xpharm1, xanharm1, xpanharm1, xharm2, xpharm2, xanharm2, xpanharm2, linewidth=0.25)
    plt.ylabel("xp", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(["Harmonique avec x(0)={}".format(xi), "Anharmonique avec x(0)={}".format(xi),
               "Harmonique avec x(0)={}".format(diff * xi), "Anharmonique avec x(0)={}".format(diff * xi)],
               loc="upper right", fontsize=8)
    plt.show()
