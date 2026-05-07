import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import time

# --- 🛡️ SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Terminal Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid Key")
        return False
    return True

# --- ⚙️ BULK DATA ENGINE ---
@st.cache_data(ttl=600)
def get_bulk_data(tickers):
    # Download all price data at once (Faster than one-by-one)
    df_all = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    return df_all

# --- 🧠 LOGIC ENGINE ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200:
            return None
           
        # Indicators
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()

        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()

        status = "🟡 HOLD"
        if price > sma200 and rsi < 35: status = "💎 BUY DIP"
        elif rvol > 2.5 and rsi < 70: status = "🔥 BREAKOUT"
        elif rsi > 78: status = "🛑 EXIT"

        stop_dist = (atr * 2)
        shares = int((funds * risk) / stop_dist) if stop_dist > 0 else 0

        return {
            "Ticker": symbol, "Price": f"${price:.2f}", "RSI": round(rsi, 1),
            "RVOL": f"{rvol:.1f}x", "Sizing": f"{shares} Shrs", "Action": status
        }
    except:
        return None

# --- 🖥️ UI ---
if check_password():
    st.set_page_config(page_title="Wealth Terminal", layout="wide")
    st.title("🐋 Institutional Wealth Terminal 2026")

    with st.sidebar:
        st.header("🕹️ Strategy Parameters")
        funds = st.number_input("Balance ($)", value=100000)
        risk = st.slider("Risk (%)", 0.5, 3.0, 1.5) / 100
       
        mode = st.selectbox("Market Feed", ["Core Watchlist", "Pre-Market Hot Picks"])
        if mode == "Core Watchlist":
            user_list = st.text_area("Tickers (Comma Separated)", "NVDA,AAPL,MSFT,TSLA,AMD,HUT,SMCI,AVGO")
        else:
            user_list = "HUT,AMD,SMCI,COMP,VEEV,PLTR,MARA,RIOT,COIN,HOOD,MSTR,SOXL"
           
        refresh = st.button("♻️ Run Scanner")

    tickers = [t.strip().upper() for t in user_list.split(",") if t]

    if refresh or "results" not in st.session_state:
        with st.spinner(f"Scanning {len(tickers)} assets..."):
            bulk_df = get_bulk_data(tickers)
            results = []
            for t in tickers:
                # Handle single vs multiple ticker dataframes
                df = bulk_df[t] if len(tickers) > 1 else bulk_df
                res = analyze_stock(t, df, funds, risk)
                if res: results.append(res)
           
            st.session_state.results = pd.DataFrame(results)
            # Heatmap Logic
            price_data = yf.download(tickers, period="6mo", progress=False)['Close']
            st.session_state.corr = price_data.corr()

    # --- DISPLAYS ---
    col1, col2 = st.columns([2, 1])
   
    with col1:
        st.subheader("📋 Market Execution")
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🔥 Risk Heatmap")
        st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
