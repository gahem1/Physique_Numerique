from numpy import *
from matplotlib import pyplot as plt
from PIL import Image
from time import time


I = complex(0,1)


class Field2D:
    def __init__(self, ds:float, wavelength:float, N:int = None, array2D:ndarray = None):
        self.dx = ds
        self.dy = ds
        self.wavelength = wavelength
        if array2D is None and N is not None:
            self.values = zeros((N, N),dtype=cdouble)
        elif array2D is not None and N is None:
            if array2D.dtype != cdouble:
                raise ValueError("Array must be complex")
            self.values = array2D
        else:
            raise ValueError("You must provide either the number of points N or an array")

    def __eq__(self, rhs):
        self.values = rhs.values
        self.dx = rhs.dx
        self.dy = rhs.dy

    @property
    def x(self) -> ndarray:
        (N, dummy) = self.values.shape
        return self.dx*linspace(-N/2, N/2, num=N, endpoint=False)

    @property
    def y(self) -> ndarray:
        (dummy, N) = self.values.shape
        return self.dy*linspace(-N/2, N/2, num=N, endpoint=False)

    def showIntensity(self):
        plt.imshow(Image.fromarray(real(self.values*conjugate(self.values))))
        plt.show()

    def showField(self):
        plt.imshow(Image.fromarray(abs(self.values)))
        plt.show()

    def showPhase(self):
        plt.imshow(Image.fromarray(angle(self.values)))
        plt.show()

    def propagate(self, distance:float):
        Efield = zeros((len(self.values[:, 0]), len(self.values[0,:])), dtype=cdouble)  # Initialize with no field for a given r
        A = self.values  # Amplitudes
        xlist, ylist = tile(self.x, (len(self.values[:, 0]), 1)).T, tile(self.y, (len(self.values[0, :]), 1))
        for i, x in enumerate(self.x):
            for j, y in enumerate(self.y):
                Ro = sqrt((x - xlist) ** 2 + (y - ylist) ** 2 + distance ** 2)
                Efield[i, j] += sum(A * exp(-I * 2 * pi * Ro / self.wavelength) / Ro)
        self.values = Efield * self.dx * self.dy

    @classmethod
    def Gaussian(self, ds:float, N:int, width:float, wavelength:float, amplitude:float = 1.0):
        allXs = ds*linspace(-N/2, N/2, num=N, endpoint=False)
        allYs = ds*linspace(-N/2, N/2, num=N, endpoint=False)

        values = zeros((N, N),dtype=cdouble)
        for (i, x) in enumerate(allXs):
            for (j, y) in enumerate(allYs):
                values[i, j] = amplitude*exp(-(x*x+y*y)/(width*width))
        return Field2D(array2D=values, ds=ds, wavelength=wavelength)


if __name__ == "__main__":
    f = Field2D.Gaussian(width=50, amplitude=32.0, ds=0.8, N=250, wavelength=2)
    f.showIntensity()
    temps = time()
    f.propagate(1000)
    print(time() - temps)
    f.showIntensity()
