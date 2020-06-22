#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nodes = []
edges = {}

with open('list.txt', 'r') as f:
    for line in f:
        i, j, w = map(int, line.split())
        nodes.append(i)
        nodes.append(j)
        edges[(i, j)] = w

nodes = list(set(nodes))

n = len(nodes)
A = np.zeros((n, n))

for i in nodes:
    for j in nodes:
        if (i, j) in edges.keys():
            A[i - 1, j - 1] = edges[(i, j)]

D = np.diag(np.sum(A, axis=1))
L = D - A

eigenval, eigenvec = np.linalg.eigh(L)

eigenvec = eigenvec.T

for i in range(len(eigenvec)):
    eigenvec[i] = eigenvec[i] / np.linalg.norm(eigenvec[i])

x = list(eigenvec[1])
y = list(eigenvec[2])
z = list(eigenvec[3])

for i in range(len(eigenvec)):
    print(i + 1, x[i], y[i], z[i])

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, marker='o', linestyle='None', color='black')
for (i, j) in edges.keys():
    ax.plot([x[i - 1], x[j - 1]], [y[i - 1], y[j - 1]], [z[i - 1], z[j - 1]],
            color='green')
# plt.show()
fig.savefig('C20.jpg')
