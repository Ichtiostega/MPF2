import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import multiprocessing as mp
import math

# 10cm x 10cm x 3cm

class FlatRadiatorModel:
    def __init__(self, dt=0.00001, dxy=0.001, dz=0.001):
        self.dxy = dxy
        self.dz = dz
        self.P = 50
        self.lbda = 200
        self.X = 0.1
        self.Y = 0.1
        self.Z = 0.03
        self.dt = dt

        self.shape = np.full((int(0.03/dz), int(0.1/dxy), int(0.1/dxy)), 20, dtype=float)

        self.K = 200 / (902 * 2700)

    def __str__(self):
        return str(self.shape)

    def iterate(self):
        #wyznaczenie podstawy
        self.shape[:][:][0] = self.shape[:][:][1] + (self.P * self.dz)/(self.lbda * self.X * self.Y)
        tmp = deepcopy(self.shape)
        dims = self.shape.shape
        for z in np.arange(1, dims[0] - 1):
            for x in np.arange(1, dims[1] - 1):
                for y in np.arange(1, dims[2] - 1):
                    tmp[z][x][y] = self.shape[z][x][y] + \
                        (self.K * self.dt)/(self.dxy**2) * (self.shape[z][x-1][y] - 2*self.shape[z][x][y] + self.shape[z][x+1][y]) + \
                        (self.K * self.dt)/(self.dxy**2) * (self.shape[z][x][y-1] - 2*self.shape[z][x][y] + self.shape[z][x][y+1]) + \
                        (self.K * self.dt)/(self.dz**2) * (self.shape[z-1][x][y] - 2*self.shape[z][x][y] + self.shape[z+1][x][y])
        self.shape = tmp

    def plot_layer(self, l):
        dims = self.shape.shape
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        GX, GY = np.meshgrid(list(range(dims[1])), list(range(dims[2])))
        ax.plot_surface(GX, GY, self.shape[:][:][l])
        plt.show()

m = FlatRadiatorModel()
for i in range(5):
    print(i+1)
    m.iterate()
m.plot_layer(2)