import numpy as np
import matplotlib.pyplot as plt


def gravity(x1, x2, x3, d):
    """The forces exerted on each body. d[0] is the distance between 1&2, d[1] between 1&3, and d[2] between 2&3"""
    ax1 = m2 * (x1 - x2) / d[0] ** 3 + m3 * (x1 - x3) / d[1] ** 3
    ax2 = m1 * (x2 - x1) / d[0] ** 3 + m3 * (x2 - x3) / d[2] ** 3
    ax3 = m1 * (x3 - x1) / d[1] ** 3 + m2 * (x3 - x2) / d[2] ** 3
    return -G * np.array([ax1, ax2, ax3])


def leapfrog(n, tf, x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3):
    t, dt = 0, tf / n

    xlist, ylist, vx, vy = np.empty([n, 3]), np.empty([n, 3]), np.empty([2, 3]), np.empty([2, 3])

    xlist[0, :], ylist[0, :] = np.array([x1, x2, x3]), np.array([y1, y2, y3])
    vx[0, :], vy[0, :] = np.array([vx1, vx2, vx3]), np.array([vy1, vy2, vy3])

    for i in range(n - 1):
        r1 = np.sqrt((xlist[i + 1, 0] - xlist[i + 1, 1]) ** 2 + (ylist[i + 1, 0] - ylist[i + 1, 1]) ** 2)
        r2 = np.sqrt((xlist[i + 1, 0] - xlist[i + 1, 2]) ** 2 + (ylist[i + 1, 0] - ylist[i + 1, 2]) ** 2)
        r3 = np.sqrt((xlist[i + 1, 1] - xlist[i + 1, 2]) ** 2 + (ylist[i + 1, 1] - ylist[i + 1, 2]) ** 2)
        dist = np.array([r1, r2, r3])

        vx[i % 2, :] += gravity(xlist[i + 1, 0], xlist[i + 1, 1], xlist[i + 1, 2], dist) * dt
        xlist[i + 2, :] = xlist[i, :] + vx[(i + 1) % 2, :] * dt
        vy[i % 2, :] += gravity(ylist[i + 1, 0], ylist[i + 1, 1], ylist[i + 1, 2], dist) * dt
        ylist[i + 2, :] = ylist[i, :] + vy[(i + 1) % 2, :] * dt

    return xlist, ylist


G = 4 * np.pi ** 2
m1 = 3
m2 = 4
m3 = 5

if __name__ == "__main__":
    N = 100000
    x1i, x2i, x3i, y1i, y2i, y3i = 1, -2, 1, 3, -1, -1
    vx1i, vx2i, vx3i, vy1i, vy2i, vy3i = 0, 0, 0, 0, 0, 0
    pointsx, pointsy = leapfrog(N, 1, x1i, x2i, x3i, y1i, y2i, y3i, vx1i, vx2i, vx3i, vy1i, vy2i, vy3i)
    plt.plot(pointsx[:, 0], pointsy[:, 0], pointsx[:, 1], pointsy[:, 1], pointsx[:, 2], pointsy[:, 2])
    plt.show()
