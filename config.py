"""
Configuration for P2-ETF-TEMPORAL-GNN engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-temporal-gnn-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Features ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Graph construction ---
CORRELATION_THRESHOLD = 0.5            # Edge if correlation > threshold
ROLLING_WINDOW = 63                    # Days for correlation estimation
NODE_FEATURE_WINDOW = 5               # Past returns per node as features

# --- TGN Model Parameters ---
HIDDEN_DIM = 64                        # GCN hidden dimension
NUM_LAYERS = 2                         # Number of GCN layers
EPOCHS = 120                            # Training epochs
BATCH_SIZE = 1                         # One graph at a time (temporal)
LEARNING_RATE = 0.001
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252                 # Minimum days of data

# --- Training scope ---
TRAIN_START = "2008-01-01"

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
