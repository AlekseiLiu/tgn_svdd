from pathlib import Path

import pandas as pd
import numpy as np

import torch

from torch_geometric.data import TemporalData



def temporal_data_from_csv(data_name, data_split, bipartite=False, attacks_names=True):
    """Load TemporalData from a CSV path resolved relative to the repo root.

    The repo root is detected based on this file location, so running the
    experiment from any working directory (e.g., `python -m src.main` from
    repo root) will work reliably.
    """

    
    repo_root = Path(__file__).resolve().parents[2]
    # path_csv = repo_root / 'data' / 'cic_2017_processing' / data_split / f'{data_name}.csv'
    path_csv = repo_root / 'data' / 'cic_2017_processing' / f'{data_name}.csv'

    df = pd.read_csv(path_csv, skiprows=1, header=None)
    
    src = torch.from_numpy(df.iloc[:, 0].values).to(torch.long)
    dst = torch.from_numpy(df.iloc[:, 1].values).to(torch.long)
    
    if bipartite:
        dst += int(src.max()) + 1
    
    t = torch.from_numpy(df.iloc[:, 2].values).to(torch.long)
    y = torch.from_numpy(df.iloc[:, 3].values).to(torch.long)
    
    # check if we have attack names in data
    y_label=None
    if attacks_names:
        y_label = df.iloc[:, 4].values
        df = df.drop(columns=4)
    
    msg = torch.from_numpy(df.iloc[:, 4:].values).to(torch.float)
    
    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)
    return data, y_label




