from graphical_models import PDAG, UndirectedGraph
from conditional_independence import CI_Tester
import itertools
import networkx as nx
from networkx.algorithms.components.connected import connected_components



# fci for DAGs 
def fci(nodes: set, ci_tester: CI_Tester, verbose=False):
    """
    CCPG algorithm
    """

    ug = nx.Graph()
    ug.add_nodes_from(nodes)
    ug.add_edges_from(itertools.combinations(nodes, 2))
    edges = set([frozenset({x, y}) for x, y in ug.edges()])

    sepset = {}

    n = 0
    while True:
        for x in [k for k in dict(ug.degree).keys() if dict(ug.degree)[k] >= n+1]:
            if ug.degree(x) < n+1: continue
            for y in list(ug.neighbors(x)):
                S = set(ug.neighbors(x)) - {y}
                for T in itertools.combinations(S, n):
                    if ci_tester.is_ci(x, y, set(T)):
                        sepset[frozenset({x, y})] = set(T)
                        edges = edges - {frozenset({x, y})}
                        ug.remove_edge(x, y)
                        break
        n += 1
        if max(dict(ug.degree).values()) < n+1:
            break

    oriented_edges = set()
    arcs = set()
    for a in nodes:
        for b in nodes - {a}:
            for c in nodes - {a, b}:
                if frozenset({a, b}) in edges and frozenset({b, c}) in edges and frozenset({a, c}) not in edges:
                    if b not in sepset[frozenset({a, c})]:
                        oriented_edges.add(frozenset({a, b}))
                        oriented_edges.add(frozenset({c, b}))
                        arcs.add((a, b))
                        arcs.add((c, b))

    cpdag = PDAG(nodes=nodes, arcs=arcs, edges=edges-oriented_edges)
    cpdag.to_complete_pdag

    return cpdag

    