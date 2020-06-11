#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()

with open('path-graph.txt', 'r') as f:
    for line in f:
        sep = line.split()
        g.add_edge(sep[0], sep[1])

# nx.draw(g)
# nx.draw_spectral(g)
# plt.show()

n = len(g.nodes)
A = np.zeros((n, n))

for i in sorted(g.nodes(), key=lambda i: int(i)):
    for j in sorted(g.nodes(), key=lambda i: int(i)):
        if (i, j) in g.edges():
            A[int(i) - 1, int(j) - 1] = 1

D = np.diag(np.sum(A, axis=1))
L = D - A

print(L)

eigenval, eigenvec = np.linalg.eigh(L)

eigenvec = eigenvec.T

for i in range(len(eigenvec)):
    eigenvec[i] = eigenvec[i] / np.linalg.norm(eigenvec[i])

x = list(eigenvec[1])
y = list(eigenvec[2])

for i in range(len(eigenvec)):
    # print(i + 1, x[i], y[i])
    print(f'{i + 1} {x[i]:.2f} {y[i]:.2f}')

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x, y, marker='o', linestyle='None', color='black')
for i, j in g.edges():
    ax.plot([x[int(i) - 1], x[int(j) - 1]], [y[int(i) - 1], y[int(j) - 1]],
            color='green')
# plt.show()
fig.savefig('path-graph.jpg')
