import numpy as np
from collections import defaultdict

def structured_by_key_indices(keys, spillover_frac=0.1, num_clients=None, seed=42):
    rng = np.random.default_rng(seed)
    keys = np.asarray(keys)
    idx_all = np.arange(len(keys))
    groups = defaultdict(list)
    for i, k in enumerate(keys): groups[k].append(i)

    all_keys = sorted(groups.keys())
    if num_clients is not None:
        all_keys = all_keys[:num_clients]

    splits = {cid: np.array(groups[k]) for cid, k in enumerate(all_keys)}

    if spillover_frac > 0:
        for cid, arr in splits.items():
            arr = arr.copy()
            rng.shuffle(arr)
            m = int(len(arr) * spillover_frac)
            if m > 0:
                keep = arr[m:]
                pool = np.setdiff1d(idx_all, arr, assume_unique=False)
                add = rng.choice(pool, size=m, replace=False) if len(pool) >= m else pool
                splits[cid] = np.concatenate([keep, add])
        splits = {k: v.tolist() for k, v in splits.items()}
    else:
        splits = {k: v.tolist() for k, v in splits.items()}
    return splits
