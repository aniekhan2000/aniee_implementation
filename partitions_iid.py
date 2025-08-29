import numpy as np

def stratified_iid_indices(labels, num_clients, seed=42, min_class_per_client=5):
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    classes = np.unique(y)
    idx_by_c = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_c[c])

    splits = {cid: [] for cid in range(num_clients)}
    for c in classes:
        parts = np.array_split(idx_by_c[c], num_clients)
        for cid in range(num_clients):
            splits[cid].extend(parts[cid].tolist())
    return splits
