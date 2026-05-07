import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v4.2", layout="wide")

# --- 2. SCRAPER ---
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

# --- 3. SECURITY ---
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

# --- 4. ANALYTICS ---
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
        price, rsi, atr = curr['Close'], curr['RSI'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.5)
        suggested_target = price + (atr * 5)
        potential_risk = price - suggested_stop
        rr_ratio = (suggested_target - price) / potential_risk if potential_risk > 0 else 0
       
        score = 0
        if price > curr['SMA200']: score += 4
        if rvol > 1.8: score += 4
        if 45 < rsi < 68: score += 2
       
        status = "🟡 MONITOR"
        if score >= 8 and rvol > 1.8: status = "🔥 ENTRY"
        elif price <= suggested_stop: status = "🛑 STOP"

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10", "R/R": f"{rr_ratio:.1f}x",
            "Entry": round(suggested_entry, 2), "Stop": round(suggested_stop, 2), "Target": round(suggested_target, 2),
            "Action": status, "Sizing": f"{int((funds * risk) / potential_risk) if potential_risk > 0 else 0} Shrs",
            "RSI": int(rsi), "News": f"https://yahoo.com{symbol}"
        }
    except: return None

# --- 5. UI ---
if check_password():
    st.title("🐋 Wealth Terminal 2026")
    with st.sidebar:
        funds = st.number_input("Portfolio $", value=100000)
        risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
        mode = st.radio("Scanner", ["Watchlist", "Hot Picks 🔥"])
        if mode == "Watchlist":
            user_input = st.text_area("Tickers", "NVDA,AAPL,TSLA,AMD,HUT,SMCI")
            t_list = [t.strip().upper() for t in user_input.split(",") if t]
        else:
            t_list = get_hot_picks()
        run = st.button("🚀 SCAN")

    # --- 6. DATA ---
    if run or "results" not in st.session_state:
        with st.spinner("Analyzing..."):
            bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
            res_list = [analyze_stock(t, bulk_df[t] if len(t_list)>1 else bulk_df, funds, risk) for t in t_list]
            st.session_state.results = pd.DataFrame([r for r in res_list if r])
            st.session_state.bulk_data = bulk_df
            # Heatmap
            hp_df = yf.download(t_list, period="6mo", progress=False)['Close']
            if isinstance(hp_df.columns, pd.MultiIndex): hp_df.columns = hp_df.columns.get_level_values(0)
            st.session_state.corr = hp_df.dropna(axis=1).corr()

    # --- 7. TABS (FIXED SECTION) ---
    tab1, tab2, tab3 = st.tabs(["📋 Execution", "📈 Indicators", "🔥 Risk"])
   
    with tab1:
        st.subheader("Market Execution")
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True, column_config={"News": st.column_config.LinkColumn("Research")})

    with tab2:
        st.subheader("Asset Charts")
        # Horizontal radio buttons are easier to tap on mobile
        sel_stock = st.radio("Select Asset:", t_list, horizontal=True)
        if sel_stock and "bulk_data" in st.session_state:
            df_ind = st.session_state.bulk_data[sel_stock] if len(t_list) > 1 else st.session_state.bulk_data
            fig = go.Figure(data=[go.Candlestick(x=df_ind.index, open=df_ind['Open'], high=df_ind['High'], low=df_ind['Low'], close=df_ind['Close'], name="Price")])
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with tab3:
        st.subheader("Portfolio Risk")
        if not st.session_state.corr.empty:
            st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
