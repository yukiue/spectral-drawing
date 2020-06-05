#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g = nx.Graph()

with open('list.txt', 'r') as f:
    for line in f:
        sep = line.split()
        g.add_edge(sep[0], sep[1], weight=sep[2])

# nx.draw(g)
# nx.draw_spectral(g)
# plt.show()

# laplacian = nx.laplacian_matrix(g)
# print(laplacian)

A = nx.to_numpy_array(g)
D = np.diag(np.sum(A, axis=1))
L = D - A

eigenval, eigenvec = np.linalg.eigh(L)

for i in range(len(eigenvec)):
    eigenvec.T[i] = eigenvec.T[i] / np.linalg.norm(eigenvec.T[i])

eigenvec = eigenvec.T

for i in range(len(eigenvec)):
    print(i, eigenvec[1][i], eigenvec[2][i], eigenvec[3][i])

x = list(eigenvec[1])
y = list(eigenvec[2])
z = list(eigenvec[3])

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, marker='o')
plt.show()
