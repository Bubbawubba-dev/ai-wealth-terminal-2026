import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import time

# --- 🛡️ SECURITY ---
# This version pulls exclusively from your Streamlit Secrets box
def check_password():
    if "password_correct" not in st.session_state:
        st.sidebar.title("🔐 Terminal Access")
        pwd = st.sidebar.text_input("Access Key", type="password")
        if st.sidebar.button("Unlock"):
            # NO hardcoded password here. It MUST be in the Secrets box.
            if pwd == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.sidebar.error("❌ Invalid Key")
        return False
    return True

# --- 📡 NOTIFICATIONS ---
def send_telegram(msg):
    # Safely fetch tokens from the vault
    token = st.secrets.get("TELEGRAM_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            url = f"https://telegram.org{token}/sendMessage?chat_id={chat_id}&text={msg}&parse_mode=Markdown"
            requests.get(url, timeout=5)
        except:
            pass

# --- ⚙️ DATA ENGINE ---
@st.cache_data(ttl=600)
def get_market_data(symbol):
    try:
        symbol = "".join(e for e in symbol if e.isalnum() or e == '.').upper()
        t = yf.Ticker(symbol)
        df = t.history(period="1y")
        if df.empty or len(df) < 200:
            return None, None
        return df, t.info
    except:
        return None, None

# --- 🧠 LOGIC ENGINE ---
def analyze(symbol, funds, risk, is_risky=False):
    df, info = get_market_data(symbol)
    if df is None:
        return {"Ticker": symbol, "Action": "❌ ERROR"}

    # Manual Math for Cloud Stability
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()

    curr = df.iloc[-1]
    price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
    rvol = curr['Volume'] / df['Volume'].tail(20).mean()
    usd_vol = (curr['Volume'] * price) / 1_000_000

    regime = "🐂 BULL" if price > sma200 else "🐻 BEAR"
    status = "🟡 HOLD"
    if regime == "🐂 BULL" and rsi < 35:
        status = "💎 BUY DIP"
    elif is_risky and rvol > 2.0 and rsi < 70:
        status = "🔥 BREAKOUT"
        send_telegram(f"🚀 *BREAKOUT:* {symbol} \nPrice: ${price:.2f}")
    elif rsi > 78:
        status = "🛑 EXIT"

    stop_dist = (atr * 2)
    shares = int((funds * risk) / stop_dist) if stop_dist > 0 else 0
    time.sleep(0.05)

    return {
        "Ticker": symbol, "Price": f"${price:.2f}", "RSI": round(rsi, 1),
        "RVOL": f"{rvol:.1f}x", "Liq": "🟢" if usd_vol > 50 else "🔴",
        "Sizing": f"{shares} Shrs", "Stop": f"${(price - stop_dist):.2f}",
        "Action": status
    }

# --- 🖥️ UI ---
if check_password():
    st.title("🐋 Institutional Wealth Terminal 2026")
    with st.sidebar:
        st.header("🕹️ Strategy Parameters")
        funds = st.number_input("Account Balance ($)", value=100000)
        risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, 1.5) / 100
        core = st.text_input("Core Holdings", "NVDA,AVGO,MSFT,REGN")
        risk_list = st.text_input("Incubator", "CIFR,NBIS,IONQ,RGTI")

    if st.button("♻️ Refresh Data"):
        st.cache_data.clear()
        all_t = [t.strip().upper() for t in (core + "," + risk_list).split(",") if t]
        data = [analyze(t, funds, risk, t in risk_list.upper()) for t in all_t]
        st.subheader("📋 Market Execution Dashboard")
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
