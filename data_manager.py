"""
Data loading and graph building for Temporal GNN engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import torch
import config

def load_master_data():
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide, tickers):
    available = [t for t in tickers if t in df_wide.columns]
    df_long = df_wide.melt(id_vars=['Date'], value_vars=available,
                           var_name='ticker', value_name='price')
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available].dropna()

def prepare_macro(df_wide):
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_temporal_graph_sequence(returns, macro):
    """
    Build a list of PyTorch Geometric `Data` objects over time.
    Each graph has:
      - x : node features (past returns + macro features)
      - edge_index : based on rolling correlation > threshold
      - y : next-day return (target for each node)
    """
    from torch_geometric.data import Data

    common = returns.index.intersection(macro.index)
    returns = returns.loc[common]
    macro = macro.loc[common]

    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    n_macro = len(macro.columns)

    # Scale returns and macro globally for node features
    ret_scaler = StandardScaler().fit(returns.values.reshape(-1, 1))
    macro_scaler = StandardScaler().fit(macro.values)

    graphs = []
    for idx in range(len(returns) - 1):
        # Node features: last NODE_FEATURE_WINDOW returns per ETF + current macro
        node_feats = []
        for tkr in tickers:
            ret_series = returns[tkr]
            if idx >= config.NODE_FEATURE_WINDOW - 1:
                window = ret_series.iloc[idx - config.NODE_FEATURE_WINDOW + 1: idx + 1].values
            else:
                window = ret_series.iloc[:idx + 1].values
                if len(window) < config.NODE_FEATURE_WINDOW:
                    window = np.pad(window, (config.NODE_FEATURE_WINDOW - len(window), 0), 'edge')
            window_scaled = ret_scaler.transform(window.reshape(-1, 1)).flatten()
            # Append current macro
            macro_vals = macro_scaler.transform(macro.iloc[idx].values.reshape(1, -1)).flatten()
            feat = np.concatenate([window_scaled, macro_vals])
            node_feats.append(feat)

        x = torch.tensor(np.stack(node_feats), dtype=torch.float32)

        # Edge index: correlation of rolling 63-day window
        if idx >= config.ROLLING_WINDOW - 1:
            rolling_ret = returns.iloc[idx - config.ROLLING_WINDOW + 1: idx + 1]
            corr = rolling_ret.corr().values
        else:
            rolling_ret = returns.iloc[:idx + 1]
            corr = rolling_ret.corr().values if len(rolling_ret) > 1 else np.eye(n_assets)

        edge_list = []
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j and corr[i, j] > config.CORRELATION_THRESHOLD:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)

        # Target: next-day return for each ETF
        y = torch.tensor(returns.iloc[idx + 1].values, dtype=torch.float32)

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs
