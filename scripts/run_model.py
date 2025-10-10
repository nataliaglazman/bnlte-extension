import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from mcmc_bspline_progress import BN_LTE_MCMC_BSpline_Optimized
from joblib import Parallel, delayed



def sklearn_normalize(data, method='standard'):
    """
    Use sklearn for more robust normalization
    """
    # Reshape to 2D: [time*patient, variable]
    n_time, n_patient, n_var = data.shape
    data_2d = data.reshape(-1, n_var)
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard', 'robust', or 'minmax'")
    
    normalized_2d = scaler.fit_transform(data_2d)
    return normalized_2d.reshape(n_time, n_patient, n_var)


bb = np.load('../data/merged_array.npy')
cols = [ 'AGE', 'PTEDUCAT', 'PTGENDER','Intracranial Volume',
       'APOE41','Hippocampus', 'Amygdala', 'Temporal Lobe',  'pT217_F', 'AB42_F', 'AB40_F', 'NfL_Q', 'GFAP_Q', 'TOTAL13']


bb = sklearn_normalize(bb, method='standard')
static_idx = [0, 1, 2]

res = []
for i in range(bb.shape[2]):
    res.append((i, -1))
    res.append((i, -2))

def run_single_chain(X, chain_id, *,
                     n_iter=10000, burnin=1000, thin=100,
                     **sampler_kwargs):
    """Run ONE BN_LTE_MCMC chain and return the list of thinned State objects."""
    sampler = BN_LTE_MCMC_BSpline_Optimized(bb, seed=chain_id, static_features=static_idx, forbidden_edges=res,
                                K_edge=5, K_baseline=5,  degree=3)
    chain   = sampler.run(n_iter=n_iter, burnin=burnin, thin=thin)
    return chain

N_CHAINS       = 4
COMMON_KWARGS  = dict(seed=1)

chains = Parallel(n_jobs=N_CHAINS)(
    delayed(run_single_chain)(
        bb, cid, n_iter=1000, burnin=100, thin=5
    )
    for cid in range(N_CHAINS)
)
print(f"Finished {N_CHAINS} chains.")