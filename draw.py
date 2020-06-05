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
nx.draw_spectral(g)
plt.show()
