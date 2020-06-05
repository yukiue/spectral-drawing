#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

eigenval, eigenvec = np.linalg.eig(L)

for i in range(len(eigenvec)):
    eigenvec[i] = eigenvec[i] / np.linalg.norm(eigenvec[i])

for i in range(nx.number_of_nodes(g)):
    print(i, eigenvec[1][i], eigenvec[2][i], eigenvec[3][i])
