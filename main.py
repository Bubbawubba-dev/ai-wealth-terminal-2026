import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal", layout="wide")

# --- 🛡️ SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Login")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid")
        return False
    return True

# --- ⚙️ DATA ---
@st.cache_data(ttl=600)
def get_data(tickers):
    return yf.download(tickers, period="1y", group_by='ticker', progress=False)

# --- 🧠 ANALYTICS ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200: return None
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        score = 0
        if price > sma200: score += 3
        if rvol > 1.8: score += 4
        if 45 < rsi < 65: score += 3
       
        status = "🟡 HOLD"
        if score >= 7 and rvol > 1.8: status = "🔥 BREAKOUT"
        elif rsi < 32: status = "💎 BUY DIP"
       
        shares = int((funds * risk) / (atr * 2)) if atr > 0 else 0
        return {"Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10", "RVOL": f"{rvol:.1f}x", "RSI": int(rsi), "Action": status, "Sizing": f"{shares} Shrs"}
    except: return None

# --- 🖥️ UI ---
if check_password():
    st.title("🐋 Wealth Terminal 2026")
    with st.sidebar:
        funds = st.number_input("Portfolio $", value=100000)
        risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
        user_input = st.text_area("Tickers", "NVDA,AAPL,TSLA,AMD,HUT,MSFT")
        run = st.button("🚀 SCAN")

    t_list = [t.strip().upper() for t in user_input.split(",") if t]
   
    if run or "results" not in st.session_state:
        with st.spinner("Scanning..."):
            bulk_df = get_data(t_list)
            res_list = [analyze_stock(t, bulk_df[t] if len(t_list)>1 else bulk_df, funds, risk) for t in t_list]
            st.session_state.results = pd.DataFrame([r for r in res_list if r])
            st.session_state.corr = yf.download(t_list, period="6mo", progress=False)['Close'].corr()

    # Display Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets", len(t_list))
    c2.metric("Market Status", "🐂 BULL" if st.session_state.results['RSI'].mean() < 70 else "🛑 HOT")
    c3.metric("Current Time", datetime.now().strftime("%H:%M"))

    st.subheader("📋 Execution Dashboard")
    st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)
   
    st.subheader("🔥 Risk Correlation")
    st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
