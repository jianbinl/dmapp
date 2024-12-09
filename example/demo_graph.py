# coding = utf-8

import numpy as np
import pandas as pd
from dmapp.graph import api


def demo_propagation_path():
    res = api.propagation_path(g, 1)
    print(f"propagation path of 0 in graph: {res}")


def demo_find_neighbors():
    who = 2
    parents = api.find_parents(g, who)
    children = api.find_children(g, who)
    print(f"parents of {who}: {parents}")
    print(f"children of {who}: {children}")


def demo_check_circle():
    cycle_list = api.check_circle(g)
    print(cycle_list)


def demo_rank_by_degree():
    out = api.rank_by_degree(g)
    print(out)


def demo_rank_by_random_walk():
    out = api.rank_by_random_walk(g)
    print(out)


if __name__ == '__main__':
    g = np.zeros((5, 5))
    g[0, 1] = 1
    g[1, 2] = 1
    g[1, 4] = 1
    g[0, 3] = 1
    g[2, 4] = 1
    g[2, 3] = 1
    g[4, 3] = 1
    g = pd.DataFrame(g, index=list('abcde'), columns=list('abcde'))
    print(g)
    x = api.find_descendants(g, 3)
    demo_rank_by_degree()
    demo_rank_by_random_walk()
    demo_check_circle()
    demo_propagation_path()
    demo_find_neighbors()
