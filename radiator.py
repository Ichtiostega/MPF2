import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import multiprocessing as mp
import math
from matplotlib import cm

# 10cm x 10cm x 6cm


class FlatRadiatorModel:
    def __init__(self, dt=0.01, dxy=0.005, dz=0.006):
        self.dxy = dxy
        self.dz = dz
        self.P = 50
        self.lbda = 200
        self.X = 0.1
        self.Y = 0.1
        self.Z = 0.06
        self.dt = dt

        self.shape = np.full((int(0.06 / dz), int(0.1 / dxy), int(0.1 / dxy)), 20, dtype=float)

        self.K = 200 / (902 * 2700)

    def __str__(self):
        return str(self.shape)

    def iterate(self):
        # wyznaczenie podstawy
        self.shape[:][:][0] = self.shape[:][:][1] + (self.P * self.dz) / (self.lbda * self.X * self.Y)
        tmp = deepcopy(self.shape)
        dims = self.shape.shape
        for z in np.arange(1, dims[0] - 1):
            for x in np.arange(1, dims[1] - 1):
                for y in np.arange(1, dims[2] - 1):
                    tmp[z][x][y] = (
                        self.shape[z][x][y]
                        + (self.K * self.dt) / (self.dxy ** 2) * (self.shape[z][x - 1][y] - 2 * self.shape[z][x][y] + self.shape[z][x + 1][y])
                        + (self.K * self.dt) / (self.dxy ** 2) * (self.shape[z][x][y - 1] - 2 * self.shape[z][x][y] + self.shape[z][x][y + 1])
                        + (self.K * self.dt) / (self.dz ** 2) * (self.shape[z - 1][x][y] - 2 * self.shape[z][x][y] + self.shape[z + 1][x][y])
                    )
        self.shape = tmp

    def plot_layer(self, l, **kwargs):
        dims = self.shape.shape
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        GX, GY = np.meshgrid(list(range(dims[1])), list(range(dims[2])))
        ax.plot_surface(GX, GY, self.shape[:][:][l], cmap=cm.coolwarm)
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.show()


class WingedRadiatorModel:
    def __init__(self, dt=0.01, dxy=0.005, dz=0.01):
        self.dxy = dxy
        self.dz = dz
        self.P = 50
        self.lbda = 200
        self.X = 0.1
        self.Y = 0.1
        self.Z = 0.1
        self.thickness = 0.02
        self.dt = dt

        self.shape = np.full((int(0.1 / dz), int(0.1 / dxy), int(0.1 / dxy)), 20, dtype=float)

        self.K = 200 / (902 * 2700)

    def __str__(self):
        return str(self.shape)

    def iterate(self):
        # wyznaczenie podstawy
        self.shape[:][:][0] = self.shape[:][:][1] + (self.P * self.dz) / (self.lbda * self.X * self.Y)
        tmp = deepcopy(self.shape)
        dims = self.shape.shape
        for z in np.arange(1, dims[0] - 1):
            for x in np.arange(1, dims[1] - 1):
                for y in np.arange(1, dims[2] - 1):
                    tmp[z][x][y] = (
                        self.shape[z][x][y]
                        + (self.K * self.dt) / (self.dxy ** 2) * (self.shape[z][x - 1][y] - 2 * self.shape[z][x][y] + self.shape[z][x + 1][y])
                        + (self.K * self.dt) / (self.dxy ** 2) * (self.shape[z][x][y - 1] - 2 * self.shape[z][x][y] + self.shape[z][x][y + 1])
                        + (self.K * self.dt) / (self.dz ** 2) * (self.shape[z - 1][x][y] - 2 * self.shape[z][x][y] + self.shape[z + 1][x][y])
                    )
        # print(int(self.thickness // self.dxy), int((self.X - self.thickness) // self.dxy))
        self.shape = tmp
        self.shape[int(self.thickness // self.dz) :,
            int(self.thickness // self.dxy) : int((self.X - self.thickness) // self.dxy),
            :] = 20


    def plot_layer(self, l, **kwargs):
        dims = self.shape.shape
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        GX, GY = np.meshgrid(list(range(dims[1])), list(range(dims[2])))
        ax.plot_surface(GX, GY, self.shape[:][:][l], cmap=cm.coolwarm)
        if 'title' in kwargs:
            plt.title(kwargs['title'])
        plt.show()


#   Flat Stability 

#       Max Time, No Min Time

# time = 0.15
# m = FlatRadiatorModel(dt=time)
# for _ in range(100):
#     m.iterate()
# print(m)
# m.plot_layer(2, title=f'dt={time}')

#       Max dxy, Min dxy

# for dxy in [0.051, 0.001]:
#     m = FlatRadiatorModel(dxy=dxy)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     m.plot_layer(2, title=f'dxy={dxy}')

#       Max dz, Min dz

# for dz in [0.03, 0.001]:
#     m = FlatRadiatorModel(dz=dz)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     m.plot_layer(1, title=f'dz={dz}')

#   Winged Stability 

#       Max Time, No Min Time

# time = 0.08
# m = WingedRadiatorModel(dt=time)
# for _ in range(100):
#     m.iterate()
# print(m)
# m.plot_layer(2, title=f'dt={time}')

#       Max dxy, Min dxy

# for dxy in [0.051, 0.001]:
#     m = WingedRadiatorModel(dxy=dxy)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     m.plot_layer(2, title=f'dxy={dxy}')

#       Max dz, Min dz

# for dz in [0.03, 0.001]:
#     m = WingedRadiatorModel(dz=dz)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     m.plot_layer(1, title=f'dz={dz}')

m = FlatRadiatorModel()
i = 0
diff = 1
while diff > 0.00001:
    tmp = m.shape
    m.iterate()
    diff = np.max(np.abs(m.shape - tmp))
    i+=1

for l in range(10):
    m.plot_layer(l, title=f'Stable layer={l}')
print(i)

m = WingedRadiatorModel()
i = 0
diff = 1
while diff > 0.00001:
    tmp = m.shape
    m.iterate()
    diff = np.max(np.abs(m.shape - tmp))
    i+=1

for l in range(10):
    m.plot_layer(l, title=f'Stable layer={l}')
print(i)