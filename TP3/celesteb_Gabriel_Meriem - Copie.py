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
    N1 = 100000
    m1 = 3
    m2 = 4
    m3 = 5
    x1i, x2i, x3i, y1i, y2i, y3i = 1, -2, 1, 3, -1, -1
    vx1i, vx2i, vx3i, vy1i, vy2i, vy3i = 0, 0, 0, 0, 0, 0
    tf1 = 1
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
    N2 = 200000
    px2, py2, vxf2, vyf2 = leapfrog(N2, 1, xi, yi, vxi, vyi, xii, yii, vxii, vyii)

    pointsx, pointsy = np.empty([N1 + N2 - 2, 3]), np.empty([N1 + N2 - 2, 3])
    pointsx[:N1 - 2, :], pointsy[:N1 - 2, :] = px1[:-2, :], py1[:-2, :]
    pointsx[N1 - 2:N1 - 2 + N2, :], pointsy[N1 - 2:N1 - 2 + N2, :] = px2, py2

    plt.plot(pointsx[:, 0], pointsy[:, 0], pointsx[:, 1], pointsy[:, 1], pointsx[:, 2], pointsy[:, 2], linewidth=0.5)
    plt.plot(pointsx[0, :], pointsy[0, :], 'r*', pointsx[N1, :], pointsy[N1, :], 'k*', markersize=5)
    plt.ylabel("y", fontsize=18)
    plt.xlabel("x", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(["m1={}".format(m1), "m2={}".format(m2), "m3={}".format(m3)], loc="upper left", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
