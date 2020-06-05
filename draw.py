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

n = len(g.nodes)
A = np.zeros((n, n))

for i in sorted(g.nodes(), key=lambda i: int(i)):
    for j in sorted(g.nodes(), key=lambda i: int(i)):
        if (i, j) in g.edges():
            A[int(i) - 1, int(j) - 1] = g[i][j]['weight']

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
    print(i, x[i], y[i], z[i])

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(x, y, z, marker='o', linestyle='None', color='black')
for i, j in g.edges():
    ax.plot([x[int(i) - 1], x[int(j) - 1]], [y[int(i) - 1], y[int(j) - 1]],
            [z[int(i) - 1], z[int(j) - 1]],
            color='green')
plt.show()
