import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- 1. BASIC CONFIG ---
st.set_page_config(page_title="Terminal", layout="wide")

# --- 2. ANALYTICS ENGINE ---
def analyze_stock(symbol, df):
    try:
        if df.empty or len(df) < 150: return None
        # Indicators
        df['SMA200'] = df['Close'].rolling(200).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
       
        curr = df.iloc[-1]
        price, rsi, sma200 = curr['Close'], curr['RSI'], curr['SMA200']
        rvol = curr['Volume'] / df['Volume'].tail(20).mean()
       
        status = "NEUTRAL"
        if price > sma200 and rsi < 35: status = "ACCUMULATE"
        elif rvol > 1.8: status = "EXPANSION"
       
        return {"TICKER": symbol, "PRICE": round(price, 2), "RSI": int(rsi), "RVOL": round(rvol, 1), "SIGNAL": status}
    except: return None

# --- 3. UI ---
st.title("INSTITUTIONAL WEALTH TERMINAL")

with st.sidebar:
    st.header("Settings")
    user_input = st.text_input("Tickers", "NVDA,AMD,HUT,SMCI,TSLA")
    run = st.button("🚀 SCAN MARKET")

if run:
    t_list = [t.strip().upper() for t in user_input.split(",") if t]
   
    with st.spinner("Fetching Data..."):
        # Data Download
        bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
       
        # Calculations
        results = []
        for t in t_list:
            df_single = bulk_df[t] if len(t_list) > 1 else bulk_df
            res = analyze_stock(t, df_single)
            if res: results.append(res)
       
        # Display Metrics
        c1, c2 = st.columns(2)
        c1.metric("Assets", len(t_list))
        c2.metric("Sync Time", datetime.now().strftime("%H:%M:%S"))
       
        # Display Table
        st.subheader("Market Intelligence")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
       
        # Display Correlation
        st.subheader("Risk Correlation")
        corr = yf.download(t_list, period="6mo", progress=False)['Close'].corr()
        st.dataframe(corr.style.background_gradient(cmap='Blues'), use_container_width=True)
else:
    st.info("Enter tickers in the sidebar and click SCAN to begin.")
