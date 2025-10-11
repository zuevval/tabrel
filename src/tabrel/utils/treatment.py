from pathlib import Path

import numpy as np
import pandas as pd


def load_ihdp_data(
    ihdp_path: Path,
) -> tuple[pd.DataFrame, list[str], str, str, str, str]:
    ihdp_cols = [s[:-1] for s in np.loadtxt(ihdp_path / "columns.txt", dtype=str)][:-2]
    ihdp_cols.extend([f"x{i}" for i in range(2, 26)])

    csvs = []
    for csv_path in (ihdp_path / "csv").glob("*.csv"):
        csvs.append(pd.read_csv(csv_path, header=None))
        break  # TODO choose a table, for now using the first table
    data = pd.concat(csvs)
    data.columns = ihdp_cols

    tau_col_name = "delta_y"
    y_fact_colname = "y_factual"
    y_cfact_colname = "y_cfactual"
    treatment_colname = "treatment"
    data[tau_col_name] = (data[y_cfact_colname] - data[y_fact_colname]) * (-1) ** data[
        "treatment"
    ]

    exclude_cols = [treatment_colname, y_cfact_colname, y_fact_colname, "mu0", "mu1"]
    return (
        data,
        exclude_cols,
        tau_col_name,
        y_fact_colname,
        y_cfact_colname,
        treatment_colname,
    )


def generate_indices(
    seed: int, n_total: int, n_query: int = 200, n_back: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    indices = np.random.permutation(n_total)
    q_indices = indices[:n_query]
    b_indices = indices[n_query:n_back]
    v_indices = indices[n_back:]
    return q_indices, b_indices, v_indices
