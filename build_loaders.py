from utils.partitions_iid import stratified_iid_indices
from utils.partitions_dirichlet import dirichlet_noniid_indices
from utils.partitions_structured import structured_by_key_indices
from utils.partition_diagnostics import save_partition_report

def build_loaders(dataset, partition, num_clients, seed=42, batch_size=64,
                  dirichlet_alpha=0.2, min_client_size=200,
                  structured_keys=None, spillover_frac=0.1,
                  diag_path=None):
    if dataset in {"mnist","cifar10"}:
        X, y = load_dataset_as_arrays(dataset)
        if partition == "iid":
            parts = stratified_iid_indices(y, num_clients, seed=seed)
        elif partition == "dirichlet":
            parts = dirichlet_noniid_indices(y, num_clients, alpha=dirichlet_alpha,
                                             seed=seed, min_client_size=min_client_size)
        else:
            raise ValueError("For images, use 'iid' or 'dirichlet' (structured needs keys).")
        if diag_path: save_partition_report(diag_path, y, parts)
        return make_pytorch_loaders(X, y, parts, batch_size=batch_size)

    # Tabular
    df = load_tabular_df(dataset)  # your function; must have df['label']
    y = df["label"].values.astype("int64")
    if partition == "iid":
        parts = stratified_iid_indices(y, num_clients, seed=seed)
    elif partition == "dirichlet":
        parts = dirichlet_noniid_indices(y, num_clients, alpha=dirichlet_alpha,
                                         seed=seed, min_client_size=min_client_size)
    elif partition in {"structured","non_iid"}:
        if structured_keys is None:
            raise ValueError("structured_keys is required for structured partition")
        parts = structured_by_key_indices(df[structured_keys].values,
                                          spillover_frac=spillover_frac,
                                          num_clients=num_clients, seed=seed)
    else:
        raise ValueError("Unsupported partition")

    if diag_path: save_partition_report(diag_path, y, parts)
    return make_pytorch_loaders_from_df(df, parts, batch_size=batch_size)
