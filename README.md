# P2-ETF-TEMPORAL-GNN + TGAT

**Temporal Graph Network & Temporal Graph Attention Network – Dynamic ETF‑Macro Graph Learning**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-TEMPORAL-GNN/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-TEMPORAL-GNN/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--temporal--gnn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-temporal-gnn-results)

## Overview

`P2-ETF-TEMPORAL-GNN` models the daily evolution of the ETF‑macro correlation graph with two complementary temporal graph models:

- **Temporal GNN (TGN)** – GCN + GRU: graph convolution captures local neighborhood information, and a gated recurrent unit tracks temporal dynamics.
- **TGAT** – Graph Attention + GRU: multi‑head self‑attention replaces GCN, allowing each ETF to directly attend to every other ETF across time.

Both models train on daily graph snapshots from 2008–2026 and predict next‑day returns. The dashboard displays their forecasts in separate tabs.

## Methodology

1. **Graph snapshots**: nodes = ETFs (features: recent returns + macro), edges = pairwise correlation > 0.5.
2. **Temporal GNN**: GCNConv + GRU updates node embeddings across time.
3. **TGAT**: TransformerConv + GRU applies multi‑head attention across the graph at each step.
4. **Prediction**: linear layer on final embedding → next‑day return for each ETF.
5. **Ranking**: top 3 ETFs per universe by predicted return.

## Dashboard

- **Temporal GNN tab**: forecasts from the GCN‑GRU model.
- **TGAT tab**: forecasts from the attention‑based model.
- Each tab has sub‑tabs for Combined, Equity Sectors, and FI/Commodities.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
