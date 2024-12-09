# coding=utf-8
import copy
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy import spatial


def check_graph(g):
    if isinstance(g, np.ndarray):
        g = pd.DataFrame(g)
    assert (g.index == g.columns).all(), f"``g`` must be 2-D np.ndarray or pd.DataFrame."
    return g


def check_circle(graph):
    """
    Check if exist circle in graph

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph

    Returns
    -------
    out: list
        if circle, return circle
    """

    graph = check_graph(graph)
    nodes = graph.index
    nx_graph = nx.DiGraph(graph.values)
    cycles = nx.simple_cycles(nx_graph)
    cycle_list = []
    for cycle in cycles:
        cycle_label = [nodes[i] for i in cycle]
        cycle_list.append(cycle_label)
    return cycle_list


def propagation_path(graph, start):
    """
    Return a propagation path from start in graph

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    start: int
        Index of graph.columns

    Returns
    -------
    result: list
        A propagation path from start in graph
    """

    graph = check_graph(graph)
    result = []
    def backtrack(path, next_):
        if len(next_) == 0:
            result.append(path.copy())
            return
        for each in next_:
            if each in path:
                continue
            path.append(each)
            new_next_ = np.where(graph.iloc[each] != 0)[0]
            backtrack(path, new_next_)
            path.pop(-1)

    next_ = np.where(graph.iloc[start] != 0)[0]
    backtrack([start], next_)
    return result


def find_neighbors(graph, who, parent: bool):
    """
    Find all parent or children node of who and return the set

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    who: int or str
        one of graph.columns.
    parent: bool
        which relationship need to find.

    Returns
    -------
    pa: set
        Index of whose parent or children.
    """

    graph = check_graph(graph)
    pa = set()
    whoi = who
    if isinstance(who, str):
        whoi = graph.columns.tolist().index(who)
    if parent:
        pa |= set(np.where(graph.iloc[:, whoi] == 1)[0])
    else:
        pa |= set(np.where(graph.iloc[whoi, :] == 1)[0])
    if isinstance(who, str):
        return {graph.columns[x] for x in pa}
    else:
        return pa


def find_parents(graph, who):
    """
    Find all parent node of each in who and return the set

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    who: int or str
        one of graph.columns.

    Returns
    -------
    pa: set
        Index of whose parent or children.
    """

    return find_neighbors(graph, who, parent=True)


def find_children(graph, who):
    """
    Find all children node of each in who and return the set

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    who: int or str
        one of graph.columns.

    Returns
    -------
    pa: set
        Index of whose parent or children.
    """

    return find_neighbors(graph, who, parent=False)


def find_descendants(graph, who):
    """
    Return all descendants of `who` not sorted

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    who: int or str
        one of graph.columns.

    Returns
    -------
    out: list
        All descendants of `who` not sorted
    """
    children = find_children(graph, who)
    des = children
    while children:
        old_des = copy.deepcopy(des)
        current_children = set()
        for child in children:
            current_children |= find_children(graph, child)
        des |= current_children
        if old_des == des:
            break
        children = current_children
    return list(des)


def node2vec_walk(graph, walk_length=5):
    """
    random walk in graph

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    walk_length: int
        length of random walk

    Returns
    -------
    out: dict
        random walk path of each node in graph
    """
    graph = check_graph(graph)
    node_vec = {}
    for node in graph.index:
        walk = [node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            current_node_children = find_children(graph, current_node)
            if len(list(current_node_children)) > 0:
                next = random.choice(list(current_node_children))
                walk.append(next)
            else:
                break
        node_vec[node] = walk
    return node_vec


def embedding_node_vector(node_vector):
    """
    embedding node vector

    Parameters
    ----------
    node_vector: dict
        random walk path in graph, out of function ``node2vec_walk``

    Returns
    -------
    out: dict
        embedded vector of each node
    """
    vec_label = list(set([y for _, x in node_vector.items() for y in x]))
    embedded_vector = {}
    for k, v in node_vector.items():
        loc = [vec_label.index(s) for s in v]
        vec = np.zeros(len(vec_label))
        vec[loc] = 1
        embedded_vector[k] = vec
    return embedded_vector


def rank_by_degree(graph, top_k=3, threshold=1e-10):
    """
    sorted by out degree

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    top_k: int
        top k root cause
    threshold: float or int
        graph[graph > threshold] = 1
        graph[graph <= threshold] = 0

    Returns
    -------
    out: pd.DataFrame
        top k root cause
    """
    graph[graph > threshold] = 1
    graph[graph <= threshold] = 0

    nds = graph.index.tolist()
    nd_path = {}
    for node in nds:
        res = find_descendants(graph, node)
        nd_path[node] = res
    nd_path = sorted(nd_path.items(), key=lambda x:len(x[1]), reverse=True)
    topk = {x: len(y) for x, y in nd_path[:top_k]}
    return pd.DataFrame.from_dict(topk, orient='index', columns=['out_degree_rank'])


def rank_by_random_walk(graph, top_k=3, walk_length=10):
    """
    sorted by effect of random walk

    Parameters
    ----------
    graph: 2-D np.ndarray or pd.DataFrame
        A graph
    top_k: int
        top k root cause
    walk_length: int
        less than graph.shape[0]
    Returns
    -------
    out: pd.DataFrame
        top k root cause
    """
    n_vec = node2vec_walk(graph, walk_length)
    n_vec = embedding_node_vector(node_vector=n_vec)

    c_graph = graph.copy()
    rows, cols = np.where(c_graph != 0)
    for r, c in zip(rows, cols):
        label_r = c_graph.columns[r]
        label_c = c_graph.columns[c]
        edge_weight = 1 - spatial.distance.cosine(n_vec[label_r], n_vec[label_c])
        c_graph.iloc[r, c] = edge_weight

    rc = {}
    for node in c_graph.columns:
        rc[node] = sum(c_graph.loc[node, :])
    rc = sorted(rc.items(), key=lambda x:x[1], reverse=True)
    top_rc = {i: j for i, j in rc[:top_k]}
    return pd.DataFrame.from_dict(top_rc, orient='index', columns=['effect_weight_rank'])
