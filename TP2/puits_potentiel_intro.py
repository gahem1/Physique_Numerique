import numpy as np
from scipy.constants import physical_constants
from matplotlib import pyplot as plt


def even(energy):
    return np.sqrt((height - energy) / energy)


def odd(energy):
    return -1 * np.sqrt(energy / (height - energy))


def general(energy):
    return np.tan(np.sqrt(omega ** 2 * mass * energy / (2 * hbar ** 2)))


hbar = physical_constants["reduced Planck constant"][0]  # [J s]
height = 20  # [eV]
mass = physical_constants["electron_mass"][0]  # [kg]
omega = 1  # [nm]
energies = np.linspace(0, 20, num=1000)  # [eV]
