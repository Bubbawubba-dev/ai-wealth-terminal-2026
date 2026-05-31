import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

st.markdown("""
<style>
.metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid")
        return False
    return True

if not check_password():
    st.stop()

# --- 3. BACKEND & DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_base_universe():
    return [
        "ASTS", "ANET", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "ARM", "ZS",
        "APP", "DPRO", "UMAC", "RKLB", "CYBR", "INTC", "CIFR", "RDDT", "QUBT",
        "QBTS", "SNOW", "HIVE", "ONDS", "F", "AVGO", "MU", "STX", "QCOM", "BE",
        "APLD", "CLSK", "CRWV", "KEEL", "CORZ", "IREN", "NBIS", "ENPH", "SMCI",
        "RGTI", "ASTC", "SHOP", "NVDA", "SHAZ", "WOLF", "AVAV", "RCAT", "KTOS", "BA"
    ]

@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=730):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, group_by="ticker", progress=False)
        return data if not data.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_fundamental_metrics(tickers):
    records = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get("marketCap")
            if cap:
                if cap >= 1e12: cap_str = f"${cap/1e12:.2f}T"
                elif cap >= 1e9: cap_str = f"${cap/1e9:.2f}B"
                elif cap >= 1e6: cap_str = f"${cap/1e6:.2f}M"
                else: cap_str = "N/A"
            else:
                cap_str = "N/A"

            margin = info.get("profitMargins")
            margin_str = f"{margin*100:.2f}%" if margin else "N/A"

            records[t] = {
                "Market Cap": cap_str,
                "P/E Ratio": info.get("trailingPE", "N/A"),
                "Profit Margin": margin_str
            }
        except:
            records[t] = {"Market Cap": "N/A", "P/E Ratio": "N/A", "Profit Margin": "N/A"}
    return records

# --- UNIFIED SIGNAL ENGINE ---
def unified_signal(df):
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    high = df["High"]
    low = df["Low"]
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    vol_ratio
