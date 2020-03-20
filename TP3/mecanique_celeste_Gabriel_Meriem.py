import numpy as np


def gravity(x1, x2, x3):
    ax1 = - G * (m2 * (x1 - x2) / np.abs(x1 - x2) ** 3 + m3 * (x1 - x3) / np.abs(x1 - x3) ** 3)
    ax2 = - G * (m1 * (x2 - x1) / np.abs(x1 - x2) ** 3 + m3 * (x2 - x3) / np.abs(x1 - x3) ** 3)
    ax3 = - G * (m1 * (x3 - x1) / np.abs(x1 - x3) ** 3 + m2 * (x3 - x2) / np.abs(x2 - x3) ** 3)
    return ax1, ax2, ax3


def leapfrog(N, tf, x1i, x2i, x3i, y1i, y2i, y3i, vx1i, vx2i, vx3i, vy1i, vy2i, vy3i):
    t, dt = 0, tf / N
    
    x1, x2, x3, y1, y2, y3 = np.empty(N), np.empty(N), np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    
    for i in range(N):
        vx1[i + 1] = vx1[i] + ax1()
    
    
G = 4 * np.pi ** 2
m1 = 3
m2 = 4
m3 = 5

if __name__ == "__main__":
    N = 10000
    x1i, x2i, x3i, y1i, y2i, y3i = 1, -2, 1, 3, -1, -1
    vx1i, vx2i, vx3i, vy1i, vy2i, vy3i = 0, 0, 0, 0, 0, 0
