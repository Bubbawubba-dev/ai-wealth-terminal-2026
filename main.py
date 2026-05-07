import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta

# --- 🛡️ SECURITY & AUTH ---
def check_password():
if "password_correct" not in st.session_state:
st.sidebar.title("🔐 Terminal Access")
pwd = st.sidebar.text_input("Access Key", type="password")
if st.sidebar.button("Unlock"):
if pwd == st.secrets.get("APP_PASSWORD", "1234"):
st.session_state["password_correct"] = True
st.rerun()
return False
return True

# --- 📡 NOTIFICATIONS ---
def send_telegram(msg):
token = st.secrets.get("TELEGRAM_TOKEN")
chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
if token and chat_id:
try:
url = f"https://telegram.org{token}/sendMessage?chat_id={chat_id}&text={msg}&parse_mode=Markdown"
requests.get(url, timeout=5)
except: pass

# --- ⚙️ DATA ENGINE ---
@st.cache_data(ttl=600)
def get_market_data(symbol):
try:
# Sanitize ticker input
symbol = "".join(e for e in symbol if e.isalnum() or e == '.').upper()
t = yf.Ticker(symbol)
df = t.history(period="1y")
if df.empty or len(df) < 200: return None, None
return df, t.info
except: return None, None

# --- 🧠 LOGIC ENGINE ---
def analyze(symbol, funds, risk, is_risky=False):
df, info = get_market_data(symbol)
if df is None: return {"Ticker": symbol, "Action": "❌ ERROR"}

# 1. Indicators
df['SMA200'] = ta.sma(df['Close'], length=200)
df['RSI'] = ta.rsi(df['Close'], length=14)
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

curr = df.iloc[-1]
price, rsi, sma200, atr = curr['Close'], curr['RSI'], curr['SMA200'], curr['ATR']

# 2. RVOL & Liquidity
rvol = curr['Volume'] / df['Volume'].tail(20).mean()
usd_vol = (curr['Volume'] * price) / 1_000_000

# 3. Strategy Logic
regime = "🐂 BULL" if price > sma200 else "🐻 BEAR"
status = "🟡 HOLD"

if regime == "🐂 BULL" and rsi < 35:
status = "💎 BUY DIP"
elif is_risky and rvol > 2.0 and rsi < 70:
status = "🔥 BREAKOUT"
send_telegram(f"🚀 *BREAKOUT:* {symbol} \nVol: {rvol:.1f}x \nPrice: ${price:.2f}")
elif rsi > 78:
status = "🛑 EXIT"

# 4. Sizing (ATR 2.0x)
stop_dist = (atr * 2)
shares = int((funds * risk) / stop_dist) if stop_dist > 0 else 0

# Small sleep to prevent rate-limiting
time.sleep(0.05)

return {
"Ticker": symbol, "Price": f"${price:.2f}", "RSI": round(rsi, 1),
"RVOL": f"{rvol:.1f}x", "Liq": "🟢" if usd_vol > 50 else "🔴",
"Sizing": f"{shares} Shrs", "Stop": f"${(price - stop_dist):.2f}",
"Action": status, "raw_close": price
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
if st.button("♻️ Refresh Data"): st.cache_data.clear()

all_t = [t.strip().upper() for t in (core + "," + risk_list).split(",") if t]
data = [analyze(t, funds, risk, t in risk_list.upper()) for t in all_t]

st.subheader("📋 Market Execution Dashboard")
st.dataframe(pd.DataFrame(data).drop(columns=['raw_close']), use_container_width=True, hide_index=True)

# Correlation Guard
if len(all_t) > 1:
st.divider()
st.subheader("🛡️ Correlation Guard (Avoid > 0.70)")
p_data = yf.download(all_t, period="6mo", progress=False)['Close']
st.dataframe(p_data.corr().style.background_gradient(cmap='RdYlGn', axis=None), use_container_width=True)

# Charts
st.divider()
sel = st.selectbox("Detailed Analysis Chart:", all_t)
if sel:
df_c, _ = get_market_data(sel)
if df_c is not None:
fig = go.Figure(data=[go.Candlestick(x=df_c.index, open=df_c['Open'], high=df_c['High'], low=df_c['Low'], close=df_c['Close'], name="Price")])
fig.add_trace(go.Scatter(x=df_c.index, y=ta.sma(df_c['Close'], 200), line=dict(color='gold', width=2), name='SMA 200'))
fig.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,b=0,t=30))
st.plotly_chart(fig, use_container_width=True)
