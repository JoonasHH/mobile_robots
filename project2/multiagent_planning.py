#!/usr/bin/env python
import numpy as np
from itertools import permutations


def time_expand(graph, nodes, obstruction):
    """Create a time-expanded graph taking into account the given
    dynamic obstruction.

    Args:
        graph (ndarray): The original graph adjacency matrix, size (n,n)
        nodes (ndarray): Node coordinates in the graph, size (n,2)
        obstruction (array-like): Indeces of obstructed nodes, length t

    Returns:
        tuple: The time-expanded graph, size (tn,tn), and node
            coordinates in the new graph, size(tn,2)
    """
    t = len(obstruction)
    n = graph.shape[0]
    expanded_graph = np.zeros((n*(t+1), n*(t+1)))
    expanded_nodes = np.zeros((n*(t+1), 2))

    for i in range(t):
        adj_t = graph.copy()
        adj_t[np.nonzero(adj_t[:, obstruction[i]]), obstruction[i]] = np.inf
        expanded_graph[i*n:i*n+n, i*n+n:i*n+2*n] = adj_t
        expanded_nodes[i*n:i*n+n] = nodes

    if t > 0:
        expanded_graph[-n:, -n:] = adj_t
    else:
        expanded_graph[-n:, -n:] = graph

    return expanded_graph, expanded_nodes
