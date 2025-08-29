import json, numpy as np

def _probs(counts):
    s = counts.sum()
    return counts / s if s > 0 else counts

def client_stats(labels, splits):
    y = np.asarray(labels)
    classes = np.unique(y)
    global_hist = np.bincount(y, minlength=classes.size)
    p_global = _probs(global_hist)

    rows, sizes, tvs, entrs = [], [], [], []
    for cid, idxs in splits.items():
        idxs = np.asarray(idxs, dtype=int)
        hist = np.bincount(y[idxs], minlength=classes.size)
        p = _probs(hist)
        tv = 0.5 * np.abs(p - p_global).sum()
        H = -(p[p>0] * np.log(p[p>0])).sum()
        rows.append({"client": int(cid), "size": int(len(idxs)),
                     "tv": float(tv), "entropy": float(H), "max_class_frac": float(p.max())})
        sizes.append(len(idxs)); tvs.append(tv); entrs.append(H)

    q_skew = float(np.std(sizes) / (np.mean(sizes) + 1e-9))
    summary = {
        "clients": len(splits),
        "size_mean": float(np.mean(sizes)), "size_cv": q_skew,
        "tv_mean": float(np.mean(tvs)), "tv_median": float(np.median(tvs)),
        "entropy_mean": float(np.mean(entrs)), "entropy_median": float(np.median(entrs))
    }
    return rows, summary

def save_partition_report(path, labels, splits):
    rows, summary = client_stats(labels, splits)
    with open(path, "w") as f:
        json.dump({"summary": summary, "clients": rows}, f, indent=2)
    return summary
