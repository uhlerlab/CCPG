import time
import numpy as np
import h5py
from causaldag import partial_correlation_suffstat, partial_correlation_test, MemoizedCI_Tester, gsp, pcalg

from data import synthetic_instance
from pc import pc_order_dep
from ccpg import ccpg

from pytorch_lightning import seed_everything
seed_everything(42)


NNODES = [5, 10, 15, 20, 50, 100]
PC_UP = 20
RERUN = 5
N_SAMPLES = 100000

for nnodes in NNODES:
    gsp_times = []
    gsp_times_plus = []
    ccpg_times = []
    if nnodes <= PC_UP:
        pcalg_times = []
        pcalg_ord_times = []
    
    model = synthetic_instance(nnodes, 1.0, True)
    for r in range(RERUN):
        samples = model.sample(N_SAMPLES)
        
        # save the matrix to HDF5
        with h5py.File(f'simulate-data/runtime/samples_{nnodes}_{r}.h5', "w") as f:
            f.create_dataset("samples", data=samples)

        # gsp
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        start_time = time.time()
        est_dag = gsp(set(range(nnodes)), ci_tester)
        end_time = time.time()
        elapsed_time = end_time - start_time
        gsp_times.append(elapsed_time)

        # gsp+
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        start_time = time.time()
        est_dag = gsp(set(range(nnodes)), ci_tester, depth=None, nruns=10)
        end_time = time.time()
        elapsed_time = end_time - start_time
        gsp_times_plus.append(elapsed_time)

        # ccpg (need to reinitialize ci_tester because it gets modified by gsp)
        suffstat = partial_correlation_suffstat(samples.T)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

        start_time = time.time()
        c, e = ccpg(set(range(nnodes)), ci_tester)
        end_time = time.time()
        elapsed_time = end_time - start_time
        ccpg_times.append(elapsed_time)

        if nnodes <= PC_UP:
            # pcalg_order_dep
            suffstat = partial_correlation_suffstat(samples.T)
            ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

            start_time = time.time()
            est_dag = pc_order_dep(set(range(nnodes)), ci_tester)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pcalg_ord_times.append(elapsed_time)

            # pcalg
            suffstat = partial_correlation_suffstat(samples.T)
            ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)

            start_time = time.time()
            est_dag = pcalg(set(range(nnodes)), ci_tester)
            end_time = time.time()
            elapsed_time = end_time - start_time
            pcalg_times.append(elapsed_time)

    print(f"----------------nnodes: {nnodes}------------------")
    print(f"gsp: {np.average(gsp_times), np.std(gsp_times)}")
    print(f"gsp+: {np.average(gsp_times_plus), np.std(gsp_times_plus)}")
    print(f"ccpg: {np.average(ccpg_times), np.std(ccpg_times)}")
    if nnodes <= PC_UP:
        print(f"pcalg_order_dep: {np.average(pcalg_ord_times), np.std(pcalg_ord_times)}")
        print(f"pcalg: {np.average(pcalg_times), np.std(pcalg_times)}")
