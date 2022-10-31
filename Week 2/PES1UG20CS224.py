"""
You can create any other helper funtions.
Do not modify the given functions
"""
import queue
import copy

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    n = len(cost)                                              
    visited = [0 for i in range(n)]                             
    fpq = queue.PriorityQueue()             
    fpq.put((heuristic[start_point], ([start_point], start_point, 0)))
    while(fpq.qsize() != 0):
        tec, node_tup = fpq.get()
        A_star_path_till_node = node_tup[0]
        node = node_tup[1]
        node_cost = node_tup[2]
        if visited[node] == 0:
            visited[node] = 1
            if node in goals:
                return A_star_path_till_node
            for neighbour_node in range(1, n):
                if cost[node][neighbour_node] > 0 and visited[neighbour_node] == 0:
                    tot_cost_node= node_cost + cost[node][neighbour_node]
                    est_tot_cost = tot_cost_node + heuristic[neighbour_node]
                    star_path_till_n = copy.deepcopy(A_star_path_till_node)
                    star_path_till_n.append(neighbour_node)
                    fpq.put((est_tot_cost, (star_path_till_n, neighbour_node, tot_cost_node)))
    return list()



def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    front = []
    n = len(cost)
    front.append(start_point)
    while len(front) != 0:
        curr = front.pop()  
        path.append(curr)
        if curr in goals:
                return path
        for i in range(n - 1, 0, -1):
                if cost[curr][i] != -1 and cost[curr][i] != 0 and (i not in path):
                        front.append(i)
    return path
