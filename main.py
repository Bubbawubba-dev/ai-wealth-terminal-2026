import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- ⚙️ CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v2.6", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for that "Dark Terminal" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; background-color: #161b22; }
    </style>
    """, unsafe_allow_index=True)

# --- 🛡️ SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Secure Login")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock Terminal"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Unauthorized Access")
        return False
    return True

# --- ⚙️ DATA ENGINE ---
@st.cache_data(ttl=600)
def get_data(tickers):
    # Bulk download for speed
    data = yf.download(tickers, period="1y", group_by='ticker', progress=False)
    return data

# --- 🧠 ANALYTICS ENGINE ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200: return None
       
        # Calculations
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()

        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()

        # --- CONVICTION SCORING ---
        score = 0
        if price > sma200: score += 3
        if rvol > 2.0: score += 4
        elif rvol > 1.2: score += 2
        if 45 < rsi < 65: score += 3
        elif rsi > 75: score -= 2

        # Filter Fakeouts
        status = "🟡 HOLD"
        if score >= 8 and rvol > 1.8: status = "🔥 BREAKOUT"
        elif rsi < 32 and price > sma200: status = "💎 BUY DIP"
        elif rsi > 80: status = "🛑 TAKE PROFIT"
        elif rvol < 1.1 and price > df['High'].shift(1).iloc[-1]: status = "⚠️ FAKEOUT"

        stop = price - (atr * 2)
        shares = int((funds * risk) / (atr * 2)) if atr > 0 else 0

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10",
            "RVOL": f"{rvol:.1f}x", "RSI": int(rsi), "Action": status,
            "Sizing": f"{shares} Shrs", "Stop": round(stop, 2)
        }
    except: return None

# --- 🖥️ MAIN INTERFACE ---
if check_password():
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://flaticon.com", width=80)
        st.header("Terminal Config")
        funds = st.number_input("Portfolio ($)", value=100000)
        risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, 1.5) / 100
       
        feed = st.selectbox("Market Feed", ["Custom Watchlist", "Pre-Market Hot Picks"])
        if feed == "Custom Watchlist":
            user_input = st.text_area("Symbols", "NVDA,AAPL,TSLA,AMD,HUT,MSFT")
        else:
            user_input = "HUT,AMD,SMCI,VEEV,COMP,PLTR,MARA,COIN,MSTR,SOXL"
       
        run = st.button("🚀 SCAN MARKET")

    # --- TOP ROW: KPI CARDS ---
    t_list = [t.strip().upper() for t in user_input.split(",") if t]
   
    if run or "results" not in st.session_state:
        with st.spinner("Crunching Big Data..."):
            bulk_df = get_data(t_list)
            res_list = []
            for t in t_list:
                df_single = bulk_df[t] if len(t_list) > 1 else bulk_df
                analysis = analyze_stock(t, df_single, funds, risk)
                if analysis: res_list.append(analysis)
           
            st.session_state.results = pd.DataFrame(res_list)
            st.session_state.corr = yf.download(t_list, period="6mo", progress=False)['Close'].corr()

    # Layout: Stats Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets Scanned", len(t_list))
    breakouts = len(st.session_state.results[st.session_state.results['Action'] == "🔥 BREAKOUT"])
    c2.metric("Breakouts Found", breakouts, delta=f"{breakouts} Alerts", delta_color="normal")
    c3.metric("Market Status", "🐂 BULLISH" if st.session_state.results['RSI'].mean() < 70 else "🛑 OVERBOUGHT")
    c4.metric("Last Update", datetime.now().strftime("%H:%M"))

    # --- MAIN CONTENT ---
    tab1, tab2 = st.tabs(["📊 Execution Dashboard", "🔥 Risk Analytics"])

    with tab1:
        st.dataframe(
            st.session_state.results.style.apply(lambda x: ['background-color: #1e3a2a' if v == '🔥 BREAKOUT' else '' for v in x], axis=1),
            use_container_width=True, hide_index=True
        )

    with tab2:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.write("### Correlation Heatmap")
            st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
        with col_right:
            st.write("### Risk Tips")
            st.info("Scores above 8/10 with RVOL > 2.0x are high-conviction entries.")
            st.warning("Avoid stocks with Correlation > 0.80 to prevent double-exposure.")
