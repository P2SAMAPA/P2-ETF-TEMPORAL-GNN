# P2-ETF-TEMPORAL-GNN

**Temporal Graph Network – Dynamic ETF‑Macro Graph Learning**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-TEMPORAL-GNN/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-TEMPORAL-GNN/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--temporal--gnn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-temporal-gnn-results)

## Overview

`P2-ETF-TEMPORAL-GNN` models the daily evolution of the ETF‑macro correlation graph. A GCN‑GRU processes a sequence of graphs (one per trading day) and predicts next‑day returns for each ETF.

## Methodology

- **Graph snapshots**: nodes = ETFs (features: recent returns + macro), edges = pairwise correlation > 0.5.
- **Temporal aggregation**: GConvGRU (GCN + GRU) updates node embeddings across time.
- **Prediction**: linear layer on final embedding → next‑day return.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
