"""
Main training script for Temporal GNN + TGAT engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from temporal_gnn_model import TGNRunner
from tgat_model import TGATRunner
import push_results

def run_temporal_gnn():
    print(f"=== P2-ETF-TEMPORAL-GNN + TGAT Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]

    macro = data_manager.prepare_macro(df_master)

    # Results containers
    tgn_all = {}
    tgat_all = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        graphs = data_manager.build_temporal_graph_sequence(returns, macro)
        if len(graphs) < config.MIN_OBSERVATIONS:
            continue

        node_feat_dim = graphs[0].x.size(1)

        # ---------- TGN ----------
        tgn = TGNRunner(node_feat_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.LEARNING_RATE, config.RANDOM_SEED)
        print(f"  Training TGN on {len(graphs)} snapshots...")
        tgn.train_sequence(graphs, epochs=config.EPOCHS)
        tgn_preds = tgn.predict_latest(graphs)

        # ---------- TGAT ----------
        tgat = TGATRunner(node_feat_dim, config.TGAT_HIDDEN_DIM, config.TGAT_NUM_HEADS,
                          config.TGAT_DROPOUT, config.TGAT_LR, config.RANDOM_SEED)
        print(f"  Training TGAT on {len(graphs)} snapshots...")
        tgat.train_sequence(graphs, epochs=config.TGAT_EPOCHS)
        tgat_preds = tgat.predict_latest(graphs)

        # Store results
        tgn_universe = {}
        tgat_universe = {}
        for i, ticker in enumerate(tickers):
            tgn_universe[ticker] = {"ticker": ticker, "forecast": float(tgn_preds[i])}
            tgat_universe[ticker] = {"ticker": ticker, "forecast": float(tgat_preds[i])}

        tgn_all[universe_name] = tgn_universe
        tgat_all[universe_name] = tgat_universe

    # Build top picks
    def build_top_picks(all_universes):
        top = {}
        for uni, data in all_universes.items():
            sorted_items = sorted(data.items(), key=lambda x: x[1]["forecast"], reverse=True)
            top[uni] = [{"ticker": t, "forecast": d["forecast"]} for t, d in sorted_items[:3]]
        return top

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "tgn": {
            "universes": tgn_all,
            "top_picks": build_top_picks(tgn_all)
        },
        "tgat": {
            "universes": tgat_all,
            "top_picks": build_top_picks(tgat_all)
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_temporal_gnn()
