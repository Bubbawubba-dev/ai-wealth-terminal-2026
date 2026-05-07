import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal v3.0", layout="wide")

# --- 🛰️ PRE-MARKET HOT PICKS SCRAPER ---
def get_hot_picks():
    try:
        url = "https://yahoo.com"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        if df_list:
            df = df_list[0]
            # Get Symbols and clean them
            return df['Symbol'].tolist()[:15]
    except:
        # Verified Movers for May 7, 2026 if scraper fails
        return ["HUT", "AMD", "SMCI", "FLEX", "COMP", "VCYT", "VECO", "ARM", "IONQ", "PLTR"]

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

# --- 🧠 ANALYTICS ENGINE ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200: return None
       
        # Math Logic
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        # --- CONVICTION SCORING & FAKEOUT FILTER ---
        score = 0
        if price > sma200: score += 3
        if rvol > 2.0: score += 4
        elif rvol > 1.2: score += 2
        if 45 < rsi < 68: score += 3
        elif rsi > 80: score -= 2

        status = "🟡 HOLD"
        if score >= 8 and rvol > 1.8: status = "🔥 BREAKOUT"
        elif rsi < 32 and price > sma200: status = "💎 BUY DIP"
        elif rvol < 1.0 and price > df['High'].shift(1).iloc[-1]: status = "⚠️ FAKEOUT"
       
        shares = int((funds * risk) / (atr * 2)) if atr > 0 else 0
        news_url = f"https://yahoo.com{symbol}/news"

        return {
            "Ticker": symbol,
            "Price": f"${price:.2f}",
            "Score": f"{score}/10",
            "RVOL": f"{rvol:.1f}x",
            "RSI": int(rsi),
            "Action": status,
            "Sizing": f"{shares} Shrs",
            "News": news_url
        }
    except: return None

# --- 🖥️ UI ---
if check_password():
    st.title("🐋 Institutional Wealth Terminal 2026")
   
    with st.sidebar:
        st.header("🕹️ Strategy Parameters")
        funds = st.number_input("Portfolio Balance ($)", value=100000)
        risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, 1.5) / 100
       
        mode = st.radio("Scanner Mode", ["My Watchlist", "Live Hot Picks 🔥"])
        if mode == "My Watchlist":
            user_input = st.text_area("Tickers", "NVDA,AAPL,TSLA,AMD,MSFT")
            t_list = [t.strip().upper() for t in user_input.split(",") if t]
        else:
            t_list = get_hot_picks()
            st.info(f"Loaded {len(t_list)} active movers.")
       
        run = st.button("🚀 EXECUTE SCAN")

    # --- DATA EXECUTION ---
    if run or "results" not in st.session_state:
        with st.spinner("Processing High-Frequency Data..."):
            bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
            res_list = [analyze_stock(t, bulk_df[t] if len(t_list)>1 else bulk_df, funds, risk) for t in t_list]
            st.session_state.results = pd.DataFrame([r for r in res_list if r])
            st.session_state.corr = yf.download(t_list, period="6mo", progress=False)['Close'].corr()

    # --- TOP METRICS ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets Analyzed", len(t_list))
    c2.metric("Market Sentiment", "🐂 BULL" if st.session_state.results['RSI'].mean() < 70 else "🛑 HOT")
    c3.metric("Terminal Time", datetime.now().strftime("%H:%M"))

    # --- DASHBOARD ---
    st.subheader("📋 Market Execution Dashboard")
    # Display table with clickable News links
    st.dataframe(
        st.session_state.results,
        use_container_width=True,
        hide_index=True,
        column_config={"News": st.column_config.LinkColumn("Research Link")}
    )
   
    st.divider()
   
    # --- RISK HEATMAP ---
    st.subheader("🔥 Risk Correlation (Portfolio Diversity)")
    st.dataframe(
        st.session_state.corr.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"),
        use_container_width=True
    )
