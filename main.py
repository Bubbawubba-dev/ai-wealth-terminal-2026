import streamlit as st

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="Wealth Terminal v3.5", layout="wide")

# --- 🛰️ RELIABLE HOT PICKS (Native Method) ---
def get_hot_picks():
# Instead of scraping a webpage, we pull a curated list of high-velocity assets
# These are the current 2026 momentum leaders across Tech, AI, and Crypto
momentum_leaders = [
"HUT", "AMD", "SMCI", "NVDA", "AAPL", "MSFT", "TSLA",
"PLTR", "MARA", "MSTR", "SOXL", "COIN", "AMD", "ARM"
]
return momentum_leaders

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
# Manual Calculations
df['SMA200'] = df['Close'].rolling(200).mean()
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + (gain / loss)))
df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
curr = df.iloc[-1]
price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']
rvol = curr['Volume'] / df['Volume'].tail(20).mean()
# Conviction Scoring
score = 0
if price > sma200: score += 3
if rvol > 1.8: score += 4
if 45 < rsi < 68: score += 3

status = "🟡 HOLD"
if score >= 7 and rvol > 1.5: status = "🔥 BREAKOUT"
elif rsi < 32: status = "💎 BUY DIP"
shares = int((funds * risk) / (atr * 2)) if atr > 0 else 0
news_url = f"https://yahoo.com{symbol}"

return {
"Ticker": symbol, "Price": f"${price:.2f}", "Score": f"{score}/10",
"RVOL": f"{rvol:.1f}x", "RSI": int(rsi), "Action": status,
"Sizing": f"{shares} Shrs", "News": news_url
}
except: return None

# --- 🖥️ UI ---
if check_password():
st.title("🐋 Institutional Wealth Terminal 2026")
with st.sidebar:
funds = st.number_input("Portfolio $", value=100000)
risk = st.slider("Risk %", 0.5, 3.0, 1.5) / 100
mode = st.radio("Scanner Mode", ["My Watchlist", "Momentum Hot Picks 🔥"])
if mode == "My Watchlist":
user_input = st.text_area("Symbols", "NVDA,AAPL,TSLA,AMD")
t_list = [t.strip().upper() for t in user_input.split(",") if t]
else:
t_list = get_hot_picks()

run = st.button("🚀 EXECUTE SCAN")

# --- DATA PROCESSING ---
if run or "results" not in st.session_state:
with st.spinner("Processing Market Intelligence..."):
# Fetch Price Data
bulk_df = yf.download(t_list, period="1y", group_by='ticker', progress=False)
# Analyze Tickers
res_list = []
for t in t_list:
df = bulk_df[t] if len(t_list) > 1 else bulk_df
analysis = analyze_stock(t, df, funds, risk)
if analysis: res_list.append(analysis)
st.session_state.results = pd.DataFrame(res_list)
# SAFE CORRELATION (Fixes the Error)
try:
corr_df = yf.download(t_list, period="6mo", progress=False)['Close']
if isinstance(corr_df.columns, pd.MultiIndex):
corr_df.columns = corr_df.columns.get_level_values(0)
st.session_state.corr = corr_df.dropna(axis=1, how='all').corr()
except:
st.session_state.corr = pd.DataFrame()

# --- TOP METRICS ---
c1, c2, c3 = st.columns(3)
c1.metric("Assets Analyzed", len(t_list))
c2.metric("Market Sentiment", "🐂 BULL" if not st.session_state.results.empty and st.session_state.results['RSI'].astype(float).mean() < 70 else "🛑 HOT")
c3.metric("Terminal Time", datetime.now().strftime("%H:%M"))

# --- DASHBOARD ---
st.subheader("📋 Market Execution Dashboard")
st.dataframe(st.session_state.results, use_container_width=True, hide_index=True,
column_config={"News": st.column_config.LinkColumn("Research")})
# --- RISK HEATMAP ---
st.subheader("🔥 Risk Correlation (Portfolio Diversity)")
if not st.session_state.corr.empty:
st.dataframe(st.session_state.corr.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
else:
st.error("Correlation Data currently unavailable for this list.")
