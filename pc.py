from graphical_models import PDAG, UndirectedGraph
from conditional_independence import CI_Tester
import itertools as itr
import networkx as nx
from networkx.algorithms.components.connected import connected_components



# order dependent pc
def pc_order_dep(nodes, ci_tester: CI_Tester=None, verbose: bool=False):
    
    nnodes = len(nodes)
    ug = UndirectedGraph(edges=set(itr.combinations(nodes, 2)))
    sepset = {}
    for c_size in range(nnodes-1):
        adjacencies = ug.neighbors
        # use the default order
        for i, j in itr.combinations(nodes, 2):
            if ug.has_edge(i, j) and len(adjacencies[i] - {j}) >= c_size:
                for cond_set in itr.combinations(adjacencies[i] - {j}, c_size):
                    if ci_tester.is_ci(i, j, cond_set):
                        if verbose: print(f"Removing {i}-{j}, separated by {cond_set}")
                        ug.delete_edge(i, j)
                        sepset[frozenset({i, j})] = cond_set
                        break

    adjacencies = ug.neighbors

    arcs = set()
    for i, k in itr.combinations(nodes, 2):
        if not ug.has_edge(i, k):
            for j in adjacencies[i] & adjacencies[k]:
                if j not in sepset[frozenset({i, k})]:
                    arcs.discard((j, k))
                    arcs.discard((j, k))
                    arcs.add((i, j))
                    arcs.add((k, j))

    cpdag = PDAG(nodes=nodes, arcs=arcs, edges=ug.edges-{frozenset({*arc}) for arc in arcs})
    cpdag.to_complete_pdag(verbose=verbose)

    return cpdag