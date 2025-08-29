import numpy as np

def dirichlet_noniid_indices(labels, num_clients, alpha=0.2, seed=42,
                             min_client_size=200, max_dominance=0.95, retry=5):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    classes = np.unique(y)
    idx_by_c = {c: rng.permutation(np.where(y == c)[0]) for c in classes}

    last = None
    for _ in range(retry):
        P = rng.dirichlet([alpha] * num_clients, size=len(classes))  # [n_classes, n_clients]
        splits = {cid: [] for cid in range(num_clients)}
        for ci, c in enumerate(classes):
            n = len(idx_by_c[c])
            take = (P[ci] / P[ci].sum() * n).astype(int)
            while take.sum() < n:
                take[rng.integers(0, num_clients)] += 1
            cuts = np.cumsum(take)[:-1]
            parts = np.split(idx_by_c[c], cuts)
            for cid in range(num_clients):
                splits[cid].extend(parts[cid].tolist())
        last = splits
        ok = True
        for cid in range(num_clients):
            sz = len(splits[cid])
            if sz < min_client_size: ok = False; break
            hist = np.bincount(y[splits[cid]], minlength=len(classes)) / max(sz, 1)
            if hist.max() > max_dominance: ok = False; break
        if ok: return splits
    return last

