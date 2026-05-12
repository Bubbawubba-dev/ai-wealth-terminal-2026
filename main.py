import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests

st.set_page_config(page_title="Wealth Terminal v4.9", layout="wide")

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

def analyze_stock(symbol, df, info, funds, risk):
    try:
        if len(df) < 200:
            return None

        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        curr = df.iloc[-1]
        price, rsi, sma50, sma200, atr = curr['Close'], curr['RSI'], curr['SMA50'], curr['SMA200'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()

        dist_from_sma50 = (price / sma50) - 1
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.5)

        score = 0
        if price > sma200: score += 3
        if rvol > 2.0: score += 4
        if 48 < rsi < 65: score += 3
        elif rsi > 75: score -= 3
        if dist_from_sma50 > 0.15: score -= 4

        status = "🟡 MONITOR"
        if score >= 8 and rvol > 1.8 and dist_from_sma50 < 0.12:
            status = "🔥 BUY"
        elif price <= suggested_stop:
            status = "🛑 STOP"

        return {
            "Ticker": symbol,
            "Price": round(price, 2),
            "Score": f"{score}/10",
            "Action": status,
            "Ext%": f"{dist_from_sma50*100:.1f}%",
            "RSI": int(rsi),
            "RVOL": f"{rvol:.1f}x",
            "Entry": round(suggested_entry, 2),
            "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs"
        }
    except:
        return None

if check_password():

    st.title("🐋 Institutional Terminal 2026")

    with st.sidebar:
        funds = st.number_input("Portfolio $", value=100000)
        risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
        mode = st.radio("Scanner", ["Watchlist", "Hot Picks 🔥"])

        t_list = (
            [t.strip().upper() for t in st.text_area("Tickers", "NVDA,AMD,HUT,SMCI,ARM").split(",") if t]
            if mode == "Watchlist"
            else get_hot_picks()
        )

        run = st.button("🚀 SCAN")

    if run or "results" not in st.session_state:
        bulk_df = yf.download(t_list, period="2y", group_by='ticker', progress=False)

        res_list = []
        for t in t_list:
            ticker_data = bulk_df[t].copy() if isinstance(bulk_df.columns, pd.MultiIndex) else bulk_df.copy()
            res = analyze_stock(t, ticker_data, yf.Ticker(t).info, funds, risk)
            if res:
                res_list.append(res)

        st.session_state.results = pd.DataFrame(res_list)
        st.session_state.bulk_data = bulk_df

    tab1, tab2 = st.tabs(["📋 Execution", "📈 Indicators"])

    with tab1:
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

    with tab2:
        sel = st.radio("Asset:", t_list, horizontal=True)

        if sel and "bulk_data" in st.session_state:
            df_raw = st.session_state.bulk_data
            df_plot = df_raw[sel].copy() if isinstance(df_raw.columns, pd.MultiIndex) else df_raw.copy()
            df_plot = df_plot.dropna()

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05
            )

            fig.add_trace(go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
