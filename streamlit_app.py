"""
Streamlit Dashboard for Temporal GNN + TGAT Engine.
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Temporal GNN+TGAT", page_icon="⏳", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("temporal_gnn_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        path = hf_hub_download(repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
                               repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def safe_pct(val):
    try:
        return f"{float(val)*100:.2f}%"
    except:
        return "N/A"

def display_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data available.")
        return
    top_picks = mode_data.get('top_picks', {})
    universes = mode_data.get('universes', {})
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    for subtab, key in zip(subtabs, keys):
        with subtab:
            top = top_picks.get(key, [])
            universe = universes.get(key, {})
            if top:
                p = top[0]
                st.markdown(f"""
                <div class="hero-card">
                    <div style="font-size: 1.2rem; opacity: 0.8;">{mode_name} TOP PICK</div>
                    <div class="hero-ticker">{p['ticker']}</div>
                    <div>Forecast: {safe_pct(p['forecast'])}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Top 3 Picks")
                rows = [{"Ticker": p['ticker'], "Forecast": safe_pct(p['forecast'])} for p in top]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("### All ETFs")
                all_rows = [{"Ticker": t, "Forecast": safe_pct(d['forecast'])} for t, d in universe.items()]
                df_all = pd.DataFrame(all_rows).sort_values("Forecast", ascending=False)
                st.dataframe(df_all, use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">⏳ P2Quant Temporal GNN + TGAT</div>', unsafe_allow_html=True)
st.markdown('<div>GCN+GRU vs Graph Attention – Temporal Graph Learning</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

main_tab1, main_tab2 = st.tabs(["🧠 Temporal GNN", "⚡ TGAT"])

with main_tab1:
    display_mode_tab(data.get('tgn'), "Temporal GNN")

with main_tab2:
    display_mode_tab(data.get('tgat'), "TGAT")
