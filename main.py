import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal v3.1", layout="wide")

# --- 🛰️ PRE-MARKET HOT PICKS SCRAPER ---
def get_hot_picks():
    try:
        url = "https://yahoo.com"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        df_list = pd.read_html(response.text)
        if df_list:
            df = df_list[0]
            return df['Symbol'].tolist()[:15]
    except:
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

# --- 🧠 ANALYTICS ENGINE (Rulebook Logic) ---
def analyze_stock(symbol, df, funds, risk):
    try:
        if df.empty or len(df) < 200: return None
       
        # 1. Indicators
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price = curr['Close']
        atr = curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        # 2. Entry / Exit / Sizing Logic
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.5) # Slightly wider stop for 2026 volatility
        suggested_target = price + (atr * 5)  # Aiming for 2x the stop distance
       
        # 3. Risk/Reward Math
        potential_risk = price - suggested_stop
        potential_reward = suggested_target - price
        rr_ratio = potential_reward / potential_risk if potential_risk > 0 else 0
       
        # 4. Conviction Scoring
        score = 0
        if price > df['SMA200'].iloc[-1]: score += 3
        if rvol > 1.8: score += 4
        if rr_ratio >= 2.0: score += 3 # Reward for better trade setups

        status = "🟡 MONITOR"
        if score >= 8 and rvol > 1.8: status = "🔥 ENTRY"
        elif price <= suggested_stop: status = "🛑 STOP"
       
        shares = int((funds * risk) / potential_risk) if potential_risk > 0 else 0

        return {
            "Ticker": symbol,
            "Price": f"${price:.2f}",
            "Score": f"{score}/10",
            "R/R Ratio": f"{rr_ratio:.1f}x", # NEW COLUMN
            "Entry": f"${suggested_entry:.2f}",
            "Stop": f"${suggested_stop:.2f}",
            "Target": f"${suggested_target:.2f}",
            "Sizing": f"{shares} Shrs",
            "Action": status,
            "News": f"https://yahoo.com{symbol}"
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

    if run or "results" not in st.session_state:
        with st.spinner("Processing High-Frequency Data..."):
            bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
            res_list = [analyze_stock(t, bulk_df[t] if len(t_list)>1 else bulk_df, funds, risk) for t in t_list]
            st.session_state.results = pd.DataFrame([r for r in res_list if r])
           
            # Clean Heatmap Data
            hp_df = yf.download(t_list, period="6mo", progress=False)['Close']
            if isinstance(hp_df.columns, pd.MultiIndex): hp_df.columns = hp_df.columns.get_level_values(0)
            st.session_state.corr = hp_df.dropna(axis=1, how='all').corr()

    c1, c2, c3 = st.columns(3)
    c1.metric("Assets Analyzed", len(t_list))
    c2.metric("Market Sentiment", "🐂 BULL" if not st.session_state.results.empty and st.session_state.results['RSI'].mean() < 70 else "🛑 HOT")
    c3.metric("Terminal Time", datetime.now().strftime("%H:%M"))

    st.subheader("📋 Market Execution Dashboard")
    st.dataframe(
        st.session_state.results,
        use_container_width=True,
        hide_index=True,
        column_config={"News": st.column_config.LinkColumn("Research Link")}
    )
   
    st.divider()
    st.subheader("🔥 Risk Correlation (Diversity Check)")
    st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
