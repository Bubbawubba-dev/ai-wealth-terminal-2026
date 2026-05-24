import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.5", layout="wide", page_icon="📈")

st.markdown("""
<style>
.metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SECURITY ---
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

if not check_password():
st.stop()

# Initialize session state for the trailing stop-loss tracking database
if "stop_loss_registry" not in st.session_state:
st.session_state.stop_loss_registry = {}

# --- 3. BACKEND & DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_base_universe():
return ["MRAM", "ASTS", "ANET", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ",
"RKLB", "SNDK", "CYBR", "INTC", "F", "PLTR", "SOUN", "BBAI", "NOW", "CIFR",
"AVGO", "MU", "STX", "LITE"]

@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=180):
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
try:
data = yf.download(tickers, start=start_date, progress=False)
if data.empty or 'Close' not in data:
return pd.DataFrame()
return data
except Exception:
return pd.DataFrame()

def calculate_momentum_metrics(df_history, tickers):
"""Quantitatively screens data with integrated Institutional Quality Filters."""
rankings = []
if df_history.empty:
return pd.DataFrame()

for ticker in tickers:
try:
close = df_history['Close'][ticker].dropna() if ticker in df_history['Close'] else pd.Series()
volume = df_history['Volume'][ticker].dropna() if ticker in df_history['Volume'] else pd.Series()
high = df_history['High'][ticker].dropna() if ticker in df_history['High'] else pd.Series()
low = df_history['Low'][ticker].dropna() if ticker in df_history['Low'] else pd.Series()

if len(close) < 20:
continue

current_price = close.iloc[-1]
current_volume = volume.iloc[-1]

# 🛡️ 1. INSTITUTIONAL QUALITY FILTERS
# Rule A: Minimum absolute share price floor
if current_price < 5.00:
continue

# Rule B: Minimum Liquidity Floor ($2,000,000 daily dollar volume traded)
daily_dollar_volume = current_price * current_volume
if daily_dollar_volume < 2000000:
continue

# Momentum calculations
perf_20d = ((current_price - close.iloc[-20]) / close.iloc[-20]) * 100
recent_vol_avg = volume.iloc[-20:-1].mean()

# Rule C: Volatility Cap to smooth out extreme outlier manipulation spikes
raw_vol_velocity = current_volume / recent_vol_avg if recent_vol_avg > 0 else 1.0
vol_velocity = min(raw_vol_velocity, 3.0)

# True Range & ATR
tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
atr_20 = tr.rolling(20).mean().iloc[-1]
current_tr = tr.iloc[-1]

atr_ratio = current_tr / atr_20 if atr_20 > 0 else 1.0
is_breakout = atr_ratio >= 1.5

rankings.append({
"Ticker": ticker,
"Price": round(current_price, 2),
"20D Return (%)": round(perf_20d, 2),
"Vol Velocity (x)": round(vol_velocity, 2),
"ATR (20)": round(atr_20, 2),
"TR/ATR Ratio": round(atr_ratio, 2),
"Explosive Flag": "🔥 BREAKOUT" if is_breakout else "Normal",
"Raw Velocity": raw_vol_velocity
})
except Exception:
continue

df_rank = pd.DataFrame(rankings)
if not df_rank.empty:
df_rank['Score'] = df_rank['20D Return (%)'] * df_rank['Vol Velocity (x)']
return df_rank.sort_values(by='Score', ascending=False).head(10).drop(columns=['Score'])
return df_rank

def calculate_sentiment_score(df_history, ticker, lookback=20):
try:
close = df_history['Close'][ticker].dropna()
high = df_history['High'][ticker].dropna()
low = df_history['Low'][ticker].dropna()

if len(close) < lookback + 1:
raise ValueError("Insufficient data.")

# RSI Component (40%)
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs.iloc[-1]))
rsi_score = np.nan_to_num(rsi, nan=50.0)

# Moving Average Extension Component (40%)
sma_20 = close.rolling(window=20).mean().iloc[-1]
current_price = close.iloc[-1]
price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

# Volatility Proxy Component (20%)
tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
atr_5 = tr.rolling(window=5).mean().iloc[-1]
atr_20 = tr.rolling(window=20).mean().iloc[-1]
vol_ratio = atr_5 / atr_20 if atr_20 > 0 else 1
vol_score = np.interp(vol_ratio, [0.8, 1.5], [80, 20])

composite_score = int(np.average([rsi_score, ma_score, vol_score], weights=[0.4, 0.4, 0.2]))

if composite_score >= 75: label = "Extreme Greed"
elif composite_score >= 55: label = "Greed"
elif composite_score >= 45: label = "Neutral"
elif composite_score >= 25: label = "Fear"
else: label = "Extreme Fear"

return {
"timestamp": datetime.now(ZoneInfo("Asia/Hong_Kong")),
"ticker": ticker,
"score": composite_score,
"label": label,
"metrics": {
"rsi_14": round(rsi_score, 1),
"ma_deviation_pct": round(price_to_sma_pct, 2),
"volatility_ratio": round(vol_ratio, 2),
"current_atr": atr_20
}
}
except Exception as e:
return {"ticker": ticker, "score": 50, "label": "Neutral", "error": str(e), "metrics": {"current_atr": 1.0}}

# --- 4. TRAILING STOP-LOSS TRACKING CONTROLLER ---
def update_trailing_stop_registry(ticker, current_price, current_atr, sentiment_label):
"""
Manages an execution matrix for risk control.
Activates trailing stops at 'Extreme Greed' using a 2x ATR cushion.
"""
registry = st.session_state.stop_loss_registry

# If sentiment cools off below Greed levels, remove tracking registry entry
if sentiment_label not in ["Extreme Greed", "Greed"]:
if ticker in registry:
del registry[ticker]
return None

# Core logic execution when asset prints extreme greed parameters
if sentiment_label == "Extreme Greed" and ticker not in registry:
initial_stop = current_price - (2 * current_atr)
registry[ticker] = {
"activation_time": datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y-%m-%d %H:%M"),
"highest_tracked_price": current_price,
"stop_loss_value": initial_stop,
"cushion_atr": current_atr
}

# Dynamic trailing adjustment loop for active trackers
if ticker in registry:
track_data = registry[ticker]
# If the price prints a new high, trail the stop loss upward
if current_price > track_data["highest_tracked_price"]:
track_data["highest_tracked_price"] = current_price
track_data["stop_loss_value"] = current_price - (2 * track_data["cushion_atr"])

# Check for trailing stop breach execution condition
if current_price <= track_data["stop_loss_value"]:
track_data["STATUS_FLAG"] = "🚨 BREACH / EXIT"
else:
track_data["STATUS_FLAG"] = "🛡️ ACTIVE HOLD"

return registry.get(ticker)

# --- 5. STREAMLIT INTERFACE LAYER ---
st.title("Wealth Terminal v12.5")

# Global data fetch pipeline execution
universe = get_base_universe()
history_df = fetch_historical_data(universe)

# Create 3 distinct structural navigation tabs
tab1, tab2, tab3 = st.tabs(["Dashboard Overview", "Breakout Engine", "Advanced Risk & Sentiment Monitoring"])

with tab1:
st.subheader("System Core Universe Status")
st.dataframe(history_df['Close'].tail(5))

with tab2:
st.subheader("Filtered Top-10 Momentum Matrix")
if not history_df.empty:
ranked_df = calculate_momentum_metrics(history_df, universe)
st.dataframe(ranked_df, use_container_width=True)
else:
st.error("Historical data engine offline.")

with tab3:
st.header("Algorithmic Risk Engine & Trailing Registry")

if not history_df.empty:
sentiment_records = []
active_tracker_rows = []

for t in universe:
sent = calculate_sentiment_score(history_df, t)
sentiment_records.append({
"Ticker": t,
"Score": sent["score"],
"Condition": sent["label"],
"RSI (14)": sent["metrics"].get("rsi_14", 50),
"MA Dev (%)": sent["metrics"].get("ma_deviation_pct", 0)
})

# Execute automated background trailing check calculations
atr_val = sent["metrics"].get("current_atr", 1.0)
p_val = history_df['Close'][t].dropna().iloc[-1] if t in history_df['Close'] else 0

tracker_output = update_trailing_stop_registry(t, p_val, atr_val, sent["label"])
if tracker_output:
active_tracker_rows.append({
"Ticker": t,
"Current Price": round(p_val, 2),
"Highest Price": round(tracker_output["highest_tracked_price"], 2),
"Stop Floor": round(tracker_output["stop_loss_value"], 2),
"Status": tracker_output["STATUS_FLAG"],
"Activated At": tracker_output["activation_time"]
})

# Layout grids
col1, col2 = st.columns(2)
with col1:
st.subheader("Synthetic Market Sentiment Grid")
st.dataframe(pd.DataFrame(sentiment_records).sort_values(by="Score", ascending=False), use_container_width=True)

with col2:
st.subheader("Extreme Greed Trailing Stop-Loss Database")
if active_tracker_rows:
st.dataframe(pd.DataFrame(active_tracker_rows), use_container_width=True)
else:
st.info("No active tickers currently meet Extreme Greed trailing stop activation protocols.")
