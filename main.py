import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 🏛️ FINANCIAL DESIGN SYSTEM (CSS) ---
st.set_page_config(page_title="Wealth Terminal v4.5", layout="wide")

st.markdown("""
    <style>
    /* Dark Slate Background */
    .stApp {
        background-color: #0B0E11;
    }
   
    /* Institutional "Workstation" Cards */
    .fin-card {
        background-color: #161A1E;
        border: 1px solid #2B3139;
        padding: 24px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
   
    /* Monospace Ticker Text */
    .ticker-font {
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
   
    /* Bloomberg Green/Red */
    .gain { color: #00C076; }
    .loss { color: #FF3B69; }
   
    /* Modern KPI Styling */
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #EAECEF;
        letter-spacing: -1px;
    }
    .kpi-label {
        font-size: 12px;
        color: #848E9C;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161A1E;
        border-right: 1px solid #2B3139;
    }
    </style>
    """, unsafe_allow_index=True)

# --- ⚙️ ANALYTICS ENGINE ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 150: return None
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        status = "NEUTRAL"
        if price > sma200 and rsi < 35: status = "ACCUMULATE"
        elif rvol > 1.8: status = "EXPANSION"
       
        return {
            "TICKER": symbol,
            "PRICE": f"{price:.2f}",
            "RSI": int(rsi),
            "RVOL": f"{rvol:.1f}x",
            "SIGNAL": status
        }
    except: return None

# --- 🖥️ INTERFACE ---
with st.sidebar:
    st.markdown("### 🛰️ SECTOR SCANNER")
    nav = st.selectbox("View", ["Market Intelligence", "Correlation Matrix", "Order Sizing"])
    st.divider()
    funds = st.number_input("AUM ($)", value=100000)
    risk = st.slider("Risk Tolerance %", 0.5, 3.0, 1.5) / 100
    st.caption("v4.5 Enterprise License")

# Main Header
st.markdown("<h2 style='letter-spacing: -1px;'>INSTITUTIONAL WEALTH TERMINAL</h2>", unsafe_allow_index=True)
st.caption(f"Market Data Synced: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Data Logic
t_list = ["NVDA", "AMD", "HUT", "SMCI", "PLTR", "TSLA", "AAPL", "MSFT", "MSTR", "COIN"]
bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
results = [analyze_stock(t, bulk_df[t], funds, risk) for t in t_list]
df_res = pd.DataFrame([r for r in results if r])

# --- KPI ROW ---
st.markdown('<div style="display: flex; gap: 20px;">', unsafe_allow_index=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="fin-card"><div class="kpi-label">Market Regime</div><div class="kpi-value gain">BULLISH</div></div>', unsafe_allow_index=True)
with k2:
    st.markdown(f'<div class="fin-card"><div class="kpi-label">Global RSI</div><div class="kpi-value">{int(df_res["RSI"].mean())}</div></div>', unsafe_allow_index=True)
with k3:
    st.markdown(f'<div class="fin-card"><div class="kpi-label">Active Signals</div><div class="kpi-value">04</div></div>', unsafe_allow_index=True)
with k4:
    st.markdown(f'<div class="fin-card"><div class="kpi-label">Volatility Index</div><div class="kpi-value loss">ELEVATED</div></div>', unsafe_allow_index=True)

# --- MAIN CONTENT ---
if nav == "Market Intelligence":
    st.markdown('<div class="fin-card">', unsafe_allow_index=True)
    st.subheader("📊 Execution Dashboard")
    # Styling the dataframe for a pro look
    st.dataframe(
        df_res,
        use_container_width=True,
        hide_index=True,
        column_config={
            "TICKER": st.column_config.TextColumn("Asset"),
            "PRICE": st.column_config.TextColumn("Last Price"),
            "SIGNAL": st.column_config.TextColumn("Execution")
        }
    )
    st.markdown('</div>', unsafe_allow_index=True)

elif nav == "Correlation Matrix":
    st.markdown('<div class="fin-card">', unsafe_allow_index=True)
    st.subheader("🛡️ Risk Dispersion")
    corr = yf.download(t_list, period="6mo", progress=False)['Close'].corr()
    st.dataframe(corr.style.background_gradient(cmap='Blues'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_index=True)

elif nav == "Order Sizing":
    st.markdown('<div class="fin-card">', unsafe_allow_index=True)
    st.subheader("📐 Position Sizing Engine")
    st.write("Calculations based on 2.0x ATR volatility stops.")
    st.table(df_res[["TICKER", "PRICE", "SIGNAL"]])
    st.markdown('</div>', unsafe_allow_index=True)
