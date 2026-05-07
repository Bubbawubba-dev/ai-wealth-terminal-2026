import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# --- 1. CONFIG ---
st.set_page_config(page_title="Wealth Terminal v4.7", layout="wide")

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
def analyze_stock(symbol, df, info, funds, risk):
    try:
        if df.empty or len(df) < 20: return None
       
        # Math
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
       
        curr = df.iloc[-1]
        price, rsi, atr = curr['Close'], curr['RSI'], curr['ATR']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        # Pivot Math
        suggested_entry = df['High'].tail(5).max()
        suggested_stop = price - (atr * 2.5)
        suggested_target = price + (atr * 5)
       
        score = 0
        if not pd.isna(curr['SMA200']) and price > curr['SMA200']: score += 4
        if rvol > 1.8: score += 4
        if 45 < rsi < 68: score += 2
       
        status = "🟡 MONITOR"
        if score >= 8: status = "🔥 BUY"
        elif price <= suggested_stop: status = "🛑 STOP"

        return {
            "Ticker": symbol, "Price": round(price, 2), "Score": f"{score}/10",
            "Action": status, "RSI": int(rsi), "RVOL": f"{rvol:.1f}x",
            "Cap($B)": f"{info.get('marketCap', 0)/1e9:.1f}B",
            "Entry": round(suggested_entry, 2), "Stop": round(suggested_stop, 2),
            "Sizing": f"{int((funds * risk)/(price - suggested_stop)) if (price-suggested_stop)>0 else 0} Shrs",
            "News": f"https://yahoo.com{symbol}"
        }
    except: return None

# --- 5. UI ---
if check_password():
    st.title("🐋 Institutional Terminal 2026")
    with st.sidebar:
        funds = st.number_input("Portfolio $", value=100000)
        risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
        mode = st.radio("Scanner", ["Watchlist", "Hot Picks 🔥"])
        t_list = [t.strip().upper() for t in st.text_area("Tickers", "NVDA,AMD,HUT,SMCI,ARM").split(",") if t] if mode == "Watchlist" else get_hot_picks()
        run = st.button("🚀 SCAN")

    if run or "results" not in st.session_state:
        bulk_df = yf.download(t_list, period="2y", group_by='ticker', progress=False)
        res_list = [analyze_stock(t, bulk_df[t], yf.Ticker(t).info, funds, risk) for t in t_list]
        st.session_state.results = pd.DataFrame([r for r in res_list if r])
        st.session_state.bulk_data = bulk_df

    tab1, tab2 = st.tabs(["📋 Execution", "📈 Indicators"])
   
    with tab1:
        st.dataframe(st.session_state.results, use_container_width=True, hide_index=True, column_config={"News": st.column_config.LinkColumn("Research")})

    with tab2:
        sel = st.radio("Asset:", t_list, horizontal=True)
        if sel and "bulk_data" in st.session_state:
            df_plot = st.session_state.bulk_data[sel]
           
            # Create Multi-Chart (Price + RSI)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
           
            # Candlesticks
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Price"), row=1, col=1)
           
            # SMA 200 (Gold) & SMA 50 (Blue)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(200).mean(), line=dict(color='gold', width=2), name='SMA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(50).mean(), line=dict(color='cyan', width=1), name='SMA 50'), row=1, col=1)
           
            # RSI Chart
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='magenta', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
           
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
