from graphical_models import PDAG, UndirectedGraph
from conditional_independence import CI_Tester
import itertools
import networkx as nx
from networkx.algorithms.components.connected import connected_components


def prefixset(nodes: set, ci_tester: CI_Tester, pset: set, verbose=False):
    """
    Outputting a larger prefix subset containting input prefix subset (pset)
    """
 
    d_set = set()
    for w in nodes - pset:
        w_in = 0
        for u in nodes:
            if w_in:
                break
            for v in nodes - pset - {w, u}:
                if ci_tester.is_ci(u, v, pset - {u}) and not ci_tester.is_ci(u, v, pset.union({w}) - {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    d_set.add(w)
                    w_in = 1
                    break

    e_set = set()
    for w in nodes - pset - d_set:
        w_in = 0
        for u in pset - {w}:
            if w_in:
                break
            for v in nodes - pset - {w, u}:
                for v_p in nodes - pset - {w, u, v}:
                    if ci_tester.is_ci(u, v_p, pset.union({v}) - {u}) and not ci_tester.is_ci(u, v_p, pset.union({w, v}) - {u}):
                        if verbose: print(f"Removing {w} from the prefix set")
                        e_set.add(w)
                        w_in = 1
                        break

    f_set = set()
    for w in nodes - pset - d_set - e_set:
        w_in = 0
        for u in pset - {w}:
            if w_in:
                break
            for v in nodes - pset - {w, u}:
                if not ci_tester.is_ci(u, v, pset - {u}) and not ci_tester.is_ci(v, w, pset) and ci_tester.is_ci(u, w, pset.union({v})- {u}):
                    if verbose: print(f"Removing {w} from the prefix set")
                    f_set.add(w)
                    w_in = 1
                    break
    
    return nodes - d_set - e_set - f_set



def ccpg(nodes: set, ci_tester: CI_Tester, verbose=False):
    """
    CCPG algorithm
    """
    
    # Step 1: get lists of prefix subsets
    s = set()
    S = []
    while s != nodes:
        s = prefixset(nodes, ci_tester, s)
        # enforce termination when ci test are not perfect
        if len(S):
            if s == S[-1] and s != nodes:
                S.append(nodes)
                break
        if verbose: print(f"Prefix set: {s}")
        S.append(s)

    # Step 2: get components of ccpg
    components = []
    for i in range(len(S)):
        edges = set()
        cond_set = S[i-1] if i > 0 else set()
        for u, v in itertools.combinations(S[i]-cond_set, 2):
            if not ci_tester.is_ci(u, v, cond_set):
                edges.add(frozenset({u, v}))
        
        ug = nx.Graph()
        ug.add_nodes_from(S[i]-cond_set)
        ug.add_edges_from(edges)
        cc = connected_components(ug)
        if verbose: print(f"Connected components: {list(cc)}")
        components.extend([set(c) for c in connected_components(ug)])        

    # Step 3: get outer component edges
    edges = set()
    for i, j in itertools.combinations(range(len(components)), 2):
        cond_set = set().union(*components[:i-1]) if i > 0 else set()
        if not set_ci(ci_tester, components[i], components[j], cond_set):
            edges.add((i,j))
    
    return components, edges



def set_ci(ci_tester: CI_Tester, set1: set, set2: set, cond_set: set):
    """
    Check if set1 is conditionally independent of set2 given cond_set
    """
    for u in set1:
        for v in set2:
            if not ci_tester.is_ci(u, v, cond_set):
                return False

    return True

