#!/usr/bin/env python
import numpy as np
from collections import deque

# Values for node status
VIRGIN = 0
ACTIVE = 1
DEAD = 2


def dijkstra(graph, start, goal):
    """Plan a path from start to goal using Dijkstra's algorithm.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node

    Returns:
        deque: Indices of nodes along the shortest path
    """
    nodes = np.int64(np.array(range(graph.shape[0])))
    nodes *= 0
    nodes[start] = ACTIVE
    cost_to_go = np.full(nodes.shape, np.inf)
    cost_to_go[start] = 0

    active_nodes = np.argwhere(nodes == ACTIVE)
    costs_an = cost_to_go[active_nodes]
    min_cost_ind = np.argmin(costs_an)
    nc = active_nodes[min_cost_ind][0]
    ncs = []
    while nc != goal:
        successors, = np.nonzero(graph[nc])
        ncs.append(nc)
        for n in successors:
            if nodes[n] != ACTIVE and nodes[n] != DEAD:
                nodes[n] = ACTIVE
                cost_to_go[n] = cost_to_go[nc] + graph[nc, n]
            elif nodes[n] == ACTIVE:
                comp = np.array([cost_to_go[n], cost_to_go[nc] + graph[nc, n]])
                cost_to_go[n] = np.min(comp)

        nodes[nc] = DEAD
        if nc == goal:
            break
        else:
            active_nodes = np.argwhere(nodes == ACTIVE)
            costs_an = cost_to_go[active_nodes]
            min_cost_ind = np.argmin(costs_an)
            nc = active_nodes[min_cost_ind][0]

    loc = goal
    path = deque([loc])
    while loc != start:
        predecessors, = np.nonzero(graph[:, loc])
        costs = cost_to_go[predecessors]
        min_cost_i = np.argmin(costs)
        loc = predecessors[min_cost_i]
        path.appendleft(loc)

    return path


def astar(graph, start, goal, heuristic):
    """Plan a path from start to goal using A* algorithm.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node
        heuristic (ndarray): The heuristic used for expanding the search

    Returns:
        deque: Indices of nodes along the shortest path
    """
    nodes = np.int64(np.array(range(graph.shape[0])))
    nodes *= 0
    nodes[start] = ACTIVE
    cost_to_go = np.full(nodes.shape, np.inf)
    cost_to_go[start] = 0
    low_app_cost = np.full(nodes.shape, np.inf)
    low_app_cost[start] = heuristic[start]

    active_nodes = np.argwhere(nodes == ACTIVE)
    app_costs_an = low_app_cost[active_nodes]
    min_cost_ind = np.argmin(app_costs_an)

    nc = active_nodes[min_cost_ind][0]
    ncs = []
    while nc != goal:
        successors, = np.nonzero(graph[nc])
        ncs.append(nc)
        for n in successors:
            if nodes[n] != ACTIVE and nodes[n] != DEAD:
                nodes[n] = ACTIVE
                cost_to_go[n] = cost_to_go[nc] + graph[nc, n]
                low_app_cost[n] = cost_to_go[n] + heuristic[n]
            elif nodes[n] == ACTIVE:
                comp = np.array([cost_to_go[n], cost_to_go[nc] + graph[nc, n]])
                cost_to_go[n] = np.min(comp)
                low_app_cost[n] = cost_to_go[n] + heuristic[n]

        nodes[nc] = DEAD
        if nc == goal:
            break
        else:
            active_nodes = np.argwhere(nodes == ACTIVE)
            app_costs_an = low_app_cost[active_nodes]
            min_cost_ind = np.argmin(app_costs_an)
            nc = active_nodes[min_cost_ind][0]

    loc = goal
    path = deque([loc])
    while loc != start:
        predecessors, = np.nonzero(graph[:, loc])
        costs = cost_to_go[predecessors]
        min_cost_i = np.argmin(costs)
        loc = predecessors[min_cost_i]
        path.appendleft(loc)

    return path


def dynamic_programming(graph, start, goal):
    """Plan a path from start to goal using dynamic programming. The goal node
    and information about the shortest paths are saved as function attributes
    to avoid unnecessary recalculation.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node

    Returns:
        deque: Indices of nodes along the shortest path
    """

    lmax = 15
    costs = np.ones((graph.shape[0], lmax + 1))
    costs *= np.inf
    policy = np.ones(graph.shape[0])
    policy *= np.nan

    for c, row in enumerate(graph):
        if c == start:
            costs, _ = optimal_cost(c, lmax, costs, policy, graph, goal)

    path = deque([start])
    loc = start
    while loc != goal:
        loc = int(policy[loc])
        path.append(loc)
    return path


def optimal_cost(c, l, costs, policy, graph, goal):
    if costs[c, l] != np.inf or l == -1:
        return costs, costs[c, l]
    successors, = np.nonzero(graph[c])
    for n in successors:
        if n == goal:
            policy[c] = n
            return costs, graph[c, n]
        else:
            costs, cost = optimal_cost(n, l - 1, costs, policy, graph, goal)
            bigj = graph[c, n] + cost
            if bigj < costs[c, 1 - l]:
                costs[c, 1 - l:] = bigj
                policy[c] = n

    return costs, costs[c, l]