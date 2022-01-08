import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import multiprocessing as mp
import math
from matplotlib import cm

# 10cm x 10cm x 6cm


class FlatRadiatorModel:
    def __init__(self, dt=0.01, dxy=0.005, dz=0.0075):
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

    def plot_layer(self, ax, l, **kwargs):
        dims = self.shape.shape
        GX, GY = np.meshgrid(list(range(dims[1])), list(range(dims[2])))
        ax.plot_surface(GX, GY, self.shape[:][:][l], cmap=cm.coolwarm)
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        ax.set_xlabel("x[cm]")
        ax.set_xticks(np.linspace(0,self.X/self.dxy, 5))
        ax.set_xticklabels(np.linspace(0,10,5))
        ax.set_ylabel("y[cm]")
        ax.set_yticks(np.linspace(0,self.X/self.dxy, 5))
        ax.set_yticklabels(np.linspace(0,10,5))
        ax.set_zlabel("T[C]")
        return ax

class WingedRadiatorModel:
    def __init__(self, dt=0.01, dxy=0.005, dz=0.0125):
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


    def plot_layer(self, ax, l, **kwargs):
        dims = self.shape.shape
        GX, GY = np.meshgrid(list(range(dims[1])), list(range(dims[2])))
        ax.plot_surface(GX, GY, self.shape[:][:][l], cmap=cm.coolwarm)
        if 'title' in kwargs:
            ax.set_title(kwargs['title'])
        ax.set_xlabel("x[cm]")
        ax.set_xticks(np.linspace(0,self.X/self.dxy, 5))
        ax.set_xticklabels(np.linspace(0,10,5))
        ax.set_ylabel("y[cm]")
        ax.set_yticks(np.linspace(0,self.X/self.dxy, 5))
        ax.set_yticklabels(np.linspace(0,10,5))
        ax.set_zlabel("T[C]")
        return ax


#   Flat Stability 

#       Max Time, No Min Time

# time = 0.15
# m = FlatRadiatorModel(dt=time)
# for _ in range(100):
#     m.iterate()
# print(m)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# m.plot_layer(ax, 2, title=f'dt={time}')
# plt.show()

#       Max dxy, Min dxy

# for dxy in [0.051, 0.001]:
#     m = FlatRadiatorModel(dxy=dxy)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     m.plot_layer(ax, 2, title=f'dxy={dxy}')
#     plt.show()

    #   Max dz, Min dz

# for dz in [0.03, 0.001]:
#     m = FlatRadiatorModel(dz=dz)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     m.plot_layer(ax, 1, title=f'dz={dz}')
#     plt.show()

#   Winged Stability 

#       Max Time, No Min Time

# time = 0.08
# m = WingedRadiatorModel(dt=time)
# for _ in range(100):
#     m.iterate()
# print(m)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# m.plot_layer(ax, 2, title=f'dt={time}')
# plt.show()

#     #   Max dxy, Min dxy

# for dxy in [0.051, 0.001]:
#     m = WingedRadiatorModel(dxy=dxy)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     m.plot_layer(ax, 2, title=f'dxy={dxy}')
#     plt.show()

    #   Max dz, Min dz

# for dz in [0.05, 0.001]:
#     m = WingedRadiatorModel(dz=dz)
#     for _ in range(100):
#         m.iterate()
#     print(m)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     m.plot_layer(ax, 1, title=f'dz={dz}')
#     plt.show()

# dz = 0.01
# m = FlatRadiatorModel(dz=dz)
# i = 0
# diff = 1
# while diff > 0.00001:
#     tmp = m.shape
#     m.iterate()
#     diff = np.max(np.abs(m.shape - tmp))
#     i+=1

# fig, axs = plt.subplots(2, 3, subplot_kw={"projection": "3d"})
# l = 0
# for row in axs:
#     for ax in row:
#         m.plot_layer(ax, l, title=f'Warstwa na {100*(l+1)*dz}cm')
#         l+=1
# plt.show()
# print(i)

dz = 0.01
m = WingedRadiatorModel(dz=dz)
i = 0
diff = 1
while diff > 0.00001:
    tmp = m.shape
    m.iterate()
    diff = np.max(np.abs(m.shape - tmp))
    i+=1

fig, axs = plt.subplots(3, 4, subplot_kw={"projection": "3d"})
l = 0
for row in axs[:2]:
    for ax in row:
        m.plot_layer(ax, l, title=f'Warstwa na {100*(l+1)*dz}cm')
        l+=1
m.plot_layer(axs[2][1], l, title=f'Warstwa na {100*(l+1)*dz}cm')
l+=1
m.plot_layer(axs[2][2], l, title=f'Warstwa na {100*(l+1)*dz}cm')
plt.show()
print(i)