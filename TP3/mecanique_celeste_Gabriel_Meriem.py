import numpy as np
import matplotlib.pyplot as plt


def gravity(x1, x2, x3):
    if x1 != x2:
        f12 = (x1 - x2) / np.abs(x1 - x2) ** 3
    else:
        f12 = 0
    if x1 != x3:
        f13 = (x1 - x3) / np.abs(x1 - x3) ** 3
    else:
        f13 = 0
    if x2 != x3:
        f23 = (x2 - x3) / np.abs(x2 - x3) ** 3
    else:
        f23 = 0
    ax1 = -G * (m2 * f12 + m3 * f13)
    ax2 = -G * (m1 * (-f12) + m3 * f23)
    ax3 = -G * (m1 * (-f13) + m2 * (-f23))
    return np.array([ax1, ax2, ax3])


def leapfrog(N, tf, x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3):
    t, dt = 0, tf / N
    
    xlist, ylist, vx, vy = np.empty([N, 3]), np.empty([N, 3]), np.array([vx1, vx2, vx3]), np.array([vy1, vy2, vy3])

    for i in range(N):
        xlist[i, :] = [x1, x2, x3]
        vx += gravity(x1, x2, x3) * dt
        x1 += vx[0] * dt
        x2 += vx[1] * dt
        x3 += vx[2] * dt
        ylist[i, :] = [y1, y2, y3]
        vy += gravity(y1, y2, y3) * dt
        y1 += vy[0] * dt
        y2 += vy[1] * dt
        y3 += vy[2] * dt
    return xlist, ylist
    
G = 4 * np.pi ** 2
m1 = 3
m2 = 4
m3 = 5

if __name__ == "__main__":
    N = 10000
    x1i, x2i, x3i, y1i, y2i, y3i = 1, -2, 1, 3, -1, -1
    vx1i, vx2i, vx3i, vy1i, vy2i, vy3i = 0, 0, 0, 0, 0, 0
    pointsx, pointsy = leapfrog(N, 1, x1i, x2i, x3i, y1i, y2i, y3i, vx1i, vx2i, vx3i, vy1i, vy2i, vy3i)
    plt.plot(pointsx[:, 0], pointsy[:, 0], pointsx[:, 1], pointsy[:, 1], pointsx[:, 2], pointsy[:, 2])
    plt.show()




