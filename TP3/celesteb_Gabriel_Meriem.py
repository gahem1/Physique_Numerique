import numpy as np
import matplotlib.pyplot as plt


def gravity(x1, x2, x3, d):
    """The forces exerted on each body. d[0] is the distance between 1&2, d[1] between 1&3, and d[2] between 2&3"""
    ax1 = m2 * (x1 - x2) / d[0] ** 3 + m3 * (x1 - x3) / d[1] ** 3
    ax2 = m1 * (x2 - x1) / d[0] ** 3 + m3 * (x2 - x3) / d[2] ** 3
    ax3 = m1 * (x3 - x1) / d[1] ** 3 + m2 * (x3 - x2) / d[2] ** 3
    return -G * np.array([ax1, ax2, ax3])


def leapfrog(n, tf, xi, yi, vxi, vyi, xii, yii, vxii, vyii):
    t, dt = 0, 2 * tf / n

    xlist, ylist, vx, vy = np.empty([n, 3]), np.empty([n, 3]), np.empty([2, 3]), np.empty([2, 3])

    xlist[0, :], ylist[0, :], vx[0, :], vy[0, :] = xi, yi, vxi, vyi
    xlist[1, :], ylist[1, :], vx[1, :], vy[1, :] = xii, yii, vxii, vyii

    for i in range(n - 2):
        r1 = np.sqrt((xlist[i + 1, 0] - xlist[i + 1, 1]) ** 2 + (ylist[i + 1, 0] - ylist[i + 1, 1]) ** 2)
        r2 = np.sqrt((xlist[i + 1, 0] - xlist[i + 1, 2]) ** 2 + (ylist[i + 1, 0] - ylist[i + 1, 2]) ** 2)
        r3 = np.sqrt((xlist[i + 1, 1] - xlist[i + 1, 2]) ** 2 + (ylist[i + 1, 1] - ylist[i + 1, 2]) ** 2)
        dist = np.array([r1, r2, r3])

        vx[i % 2, :] += gravity(xlist[i + 1, 0], xlist[i + 1, 1], xlist[i + 1, 2], dist) * dt
        xlist[i + 2, :] = xlist[i, :] + vx[(i + 1) % 2, :] * dt
        vy[i % 2, :] += gravity(ylist[i + 1, 0], ylist[i + 1, 1], ylist[i + 1, 2], dist) * dt
        ylist[i + 2, :] = ylist[i, :] + vy[(i + 1) % 2, :] * dt

    return xlist, ylist, vx, vy


G = 4 * np.pi ** 2

if __name__ == "__main__":
    N1 = 280000
    m1 = 3
    m2 = 4
    m3 = 5
    x1i, x2i, x3i, y1i, y2i, y3i = 1, -2, 1, 3, -1, -1
    vx1i, vx2i, vx3i, vy1i, vy2i, vy3i = 0, 0, 0, 0, 0, 0
    tf1 = 2.5
    step = tf1 * 2 / N1

    xi, yi = np.array([x1i, x2i, x3i]), np.array([y1i, y2i, y3i])
    vxi, vyi = np.array([vx1i, vx2i, vx3i]), np.array([vy1i, vy2i, vy3i])

    r1 = np.sqrt((x1i - x2i) ** 2 + (y1i - y2i) ** 2)
    r2 = np.sqrt((x1i - x3i) ** 2 + (y1i - y3i) ** 2)
    r3 = np.sqrt((x2i - x3i) ** 2 + (y2i - y3i) ** 2)
    dist = np.array([r1, r2, r3])

    rx1 = 0.25 * step * vxi
    ry1 = 0.25 * step * vyi
    kx1 = 0.25 * step * gravity(x1i, x2i, x3i, dist)
    ky1 = 0.25 * step * gravity(y1i, y2i, y3i, dist)

    r1 = np.sqrt((x1i + rx1[0] - x2i - rx1[1]) ** 2 + (y1i + ry1[0] - y2i - ry1[1]) ** 2)
    r2 = np.sqrt((x1i + rx1[0] - x3i - rx1[2]) ** 2 + (y1i + ry1[0] - y3i - ry1[2]) ** 2)
    r3 = np.sqrt((x2i + rx1[1] - x3i - rx1[2]) ** 2 + (y2i + ry1[1] - y3i - ry1[2]) ** 2)
    dist = np.array([r1, r2, r3])

    xii = xi + 0.5 * step * (vxi + kx1)
    yii = yi + 0.5 * step * (vyi + ky1)
    vxii = vxi + 0.5 * step * gravity(x1i + rx1[0], x2i + rx1[1], x3i + rx1[2], dist)
    vyii = vyi + 0.5 * step * gravity(y1i + ry1[0], y2i + ry1[1], y3i + ry1[2], dist)

    px1, py1, vxf1, vyf1 = leapfrog(N1, tf1, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px1[-2, :], px1[-1, :], py1[-2, :], py1[-1, :]
    vxi, vxii, vyi, vyii = vxf1[-2, :], vxf1[-1, :], vyf1[-2, :], vyf1[-1, :]
    N2 = 300000
    px2, py2, vxf2, vyf2 = leapfrog(N2, 0.25, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px2[-2, :], px2[-1, :], py2[-2, :], py2[-1, :]
    vxi, vxii, vyi, vyii = vxf2[-2, :], vxf2[-1, :], vyf2[-2, :], vyf2[-1, :]
    N3 = 150000
    px3, py3, vxf3, vyf3 = leapfrog(N3, 0.89, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px3[-2, :], px3[-1, :], py3[-2, :], py3[-1, :]
    vxi, vxii, vyi, vyii = vxf3[-2, :], vxf3[-1, :], vyf3[-2, :], vyf3[-1, :]
    N4 = 1000000
    px4, py4, vxf4, vyf4 = leapfrog(N4, 0.01, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px4[-2, :], px4[-1, :], py4[-2, :], py4[-1, :]
    vxi, vxii, vyi, vyii = vxf4[-2, :], vxf4[-1, :], vyf4[-2, :], vyf4[-1, :]
    N5 = 30000
    px5, py5, vxf5, vyf5 = leapfrog(N5, 0.25, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px5[-2, :], px5[-1, :], py5[-2, :], py5[-1, :]
    vxi, vxii, vyi, vyii = vxf5[-2, :], vxf5[-1, :], vyf5[-2, :], vyf5[-1, :]
    N6 = 800000
    px6, py6, vxf6, vyf6 = leapfrog(N6, 0.02, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px6[-2, :], px6[-1, :], py6[-2, :], py6[-1, :]
    vxi, vxii, vyi, vyii = vxf6[-2, :], vxf6[-1, :], vyf6[-2, :], vyf6[-1, :]
    N7 = 200000
    px7, py7, vxf7, vyf7 = leapfrog(N7, 0.46, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px7[-2, :], px7[-1, :], py7[-2, :], py7[-1, :]
    vxi, vxii, vyi, vyii = vxf7[-2, :], vxf7[-1, :], vyf7[-2, :], vyf7[-1, :]
    N8 = 800000
    px8, py8, vxf8, vyf8 = leapfrog(N8, 0.02, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px8[-2, :], px8[-1, :], py8[-2, :], py8[-1, :]
    vxi, vxii, vyi, vyii = vxf8[-2, :], vxf8[-1, :], vyf8[-2, :], vyf8[-1, :]
    N9 = 100000
    px9, py9, vxf9, vyf9 = leapfrog(N9, 0.30, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    xi, xii, yi, yii = px9[-2, :], px9[-1, :], py9[-2, :], py9[-1, :]
    vxi, vxii, vyi, vyii = vxf9[-2, :], vxf9[-1, :], vyf9[-2, :], vyf9[-1, :]
    N10 = 500000
    px10, py10, vxf10, vyf10 = leapfrog(N10, 0.001, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    Nt2 = N1 + N2
    Nt3 = Nt2 + N3
    Nt4 = Nt3 + N4
    Nt5 = Nt4 + N5
    Nt6 = Nt5 + N6
    Nt7 = Nt6 + N7
    Nt8 = Nt7 + N8
    Nt9 = Nt8 + N9
    NT = Nt9 + N10

    pointsx, pointsy = np.empty([NT - 2, 3]), np.empty([NT - 2, 3])
    pointsx[:N1 - 2, :], pointsy[:N1 - 2, :] = px1[:-2, :], py1[:-2, :]
    pointsx[N1 - 2:Nt2 - 4, :], pointsy[N1 - 2:Nt2 - 4, :] = px2[:-2, :], py2[:-2, :]
    pointsx[Nt2 - 4:Nt3 - 6, :], pointsy[Nt2 - 4:Nt3 - 6, :] = px3[:-2, :], py3[:-2, :]
    pointsx[Nt3 - 6:Nt4 - 8, :], pointsy[Nt3 - 6:Nt4 - 8, :] = px4[:-2, :], py4[:-2, :]
    pointsx[Nt4 - 8:Nt5 - 10, :], pointsy[Nt4 - 8:Nt5 - 10, :] = px5[:-2, :], py5[:-2, :]
    pointsx[Nt5 - 10:Nt6 - 12, :], pointsy[Nt5 - 10:Nt6 - 12, :] = px6[:-2, :], py6[:-2, :]
    pointsx[Nt6 - 12:Nt7 - 14, :], pointsy[Nt6 - 12:Nt7 - 14, :] = px7[:-2, :], py7[:-2, :]
    pointsx[Nt7 - 14:Nt8 - 16, :], pointsy[Nt7 - 14:Nt8 - 16, :] = px8[:-2, :], py8[:-2, :]
    pointsx[Nt8 - 16:Nt9 - 18, :], pointsy[Nt8 - 16:Nt9 - 18, :] = px9[:-2, :], py9[:-2, :]
    pointsx[Nt9 - 18:NT - 18, :], pointsy[Nt9 - 18:NT - 18, :] = px10, py10

    plt.plot(pointsx[::50, 0], pointsy[::50, 0], pointsx[::50, 1], pointsy[::50, 1], pointsx[::50, 2],
             pointsy[::50, 2], linewidth=0.5)
    plt.plot(pointsx[0, :], pointsy[0, :], 'r*', markersize=5)
    plt.ylabel("y", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["m1={}".format(m1), "m2={}".format(m2), "m3={}".format(m3)], loc="upper left", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
