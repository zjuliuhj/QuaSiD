from pathlib import Path

import numpy as np
import pandas as pd


def append_metrics_csv(csv_path, row):
    """Append one experiment metric row to its local CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        output = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        output = new_row
    output.to_csv(csv_path, index=False, float_format="%.5f")
    return str(csv_path)


def update_total_metrics_csv(csv_path, row, key_columns=("experiment", "milestone")):
    """Replace the matching dataset-level metrics row and keep total.csv sorted."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])

    if csv_path.exists():
        output = pd.read_csv(csv_path)
        for column in new_row.columns:
            if column not in output.columns:
                output[column] = np.nan
        for column in output.columns:
            if column not in new_row.columns:
                new_row[column] = np.nan

        output = output[list(new_row.columns)]
        mask = pd.Series(True, index=output.index)
        for column in key_columns:
            if column in output.columns and column in new_row.columns:
                mask &= output[column].astype(str) == str(new_row.iloc[0][column])
        output = output.loc[~mask]
        output = pd.concat([output, new_row], ignore_index=True, sort=False)
    else:
        output = new_row

    sort_columns = [
        column
        for column in ("seq_length", "backbone", "use_pmi_corr", "experiment")
        if column in output.columns
    ]
    if sort_columns:
        output = output.sort_values(sort_columns).reset_index(drop=True)
    output.to_csv(csv_path, index=False, float_format="%.5f")
    return str(csv_path)
