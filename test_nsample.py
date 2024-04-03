from pytorch_lightning import seed_everything
seed_everything(42)
import numpy as np
import h5py

from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp, pcalg
from pc import pc_order_dep
from ccpg import ccpg
from data import synthetic_instance


nnodes = 10
model = synthetic_instance(nnodes, 1.0, True)
print(model.DAG.arcs)

for r in range(5):
    # save a bunch of samples for Rstudio
    for nsample in np.arange(100, 5000, 100):
        samples = model.sample(nsample)
        with h5py.File(f'simulate-data/nsample/samples_{nsample}_{r}.h5', "w") as f:
            f.create_dataset("samples", data=samples)
            
    print(f"-------------Run {r}----------------")
    # ccpg
    nsample = 100

    while True:
        samples = model.sample(nsample)
        nsample += 100
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        c, e = ccpg(set(range(nnodes)), ci_tester, verbose=False)

        if max([len(i) for i in c]) > 1:
            continue
        else:
            edges = set()
            for (i,j) in e:
                edges.add((list(c[i])[0],list(c[j])[0]))
            if edges == model.DAG.arcs:
                break

    print('ccpg', nsample)


    # pc_order_dep
    nsample = 100

    while True:
        samples = model.sample(nsample)
        nsample += 100
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        est_dag = pc_order_dep(set(range(nnodes)), ci_tester)
        if est_dag.arcs == model.DAG.arcs and est_dag.edges == set():
            break

    print('pc_order_dep', nsample)

    # pc
    nsample = 100

    while True:
        samples = model.sample(nsample)
        nsample += 100
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        est_dag = pcalg(set(range(nnodes)), ci_tester)
        if est_dag.arcs == model.DAG.arcs and est_dag.edges == set():
            break

    print('pc', nsample)

    # gsp
    nsample = 100

    while True:
        samples = model.sample(nsample)
        nsample += 100
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        est_dag = gsp(set(range(nnodes)), ci_tester)
        if est_dag.arcs == model.DAG.arcs:
            break

    print('gsp', nsample)

    # gsp+
    nsample = 100

    while True:
        samples = model.sample(nsample)
        nsample += 100
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        est_dag = gsp(set(range(nnodes)), ci_tester, depth=None, nruns=10)
        if est_dag.arcs == model.DAG.arcs:
            break

    print('gsp+', nsample)

