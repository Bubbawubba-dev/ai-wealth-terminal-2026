import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

# Custom CSS to improve terminal UI scannability
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

# --- 3. BACKEND & DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_base_universe():
    """Returns a stable, responsive base core universe of volatile/momentum equities."""
    return ["MRAM", "ASTS", "ANET", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ",
    "RKLB", "SNDK", "CYBR", "INTC", "F", "PLTR", "SOUN", "BBAI", "NOW", "CIFR",
    "AVGO", "MU", "STX", "LITE"]

@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=730): # 730 days lookback needed for accurate 200D SMA metrics
"""Safely fetches multi-ticker daily historical data across the core universe."""
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
try:
data = yf.download(tickers, start=start_date, progress=False)
if data.empty or 'Close' not in data:
return pd.DataFrame()
return data
except Exception:
return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_fundamental_metrics(tickers):
"""Fetches high-latency corporate fundamental metrics from yfinance.info. Highly cached."""
fundamental_records = {}
for ticker in tickers:
try:
t_obj = yf.Ticker(ticker)
info = t_obj.info

# Formulating short human readable market cap strings
raw_cap = info.get("marketCap", None)
if raw_cap and raw_cap >= 1e12:
cap_str = f"${raw_cap / 1e12:.2f}T"
elif raw_cap and raw_cap >= 1e9:
cap_str = f"${raw_cap / 1e9:.2f}B"
elif raw_cap and raw_cap >= 1e6:
cap_str = f"${raw_cap / 1e6:.2f}M"
else:
cap_str = "N/A"

# Parse and scale margins safely
margin_raw = info.get("profitMargins", None)
margin_pct = f"{margin_raw * 100:.2f}%" if margin_raw is not None else "N/A"

fundamental_records[ticker] = {
"Market Cap": cap_str,
"P/E Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
"Profit Margin": margin_pct
}
except Exception:
fundamental_records[ticker] = {"Market Cap": "N/A", "P/E Ratio": "N/A", "Profit Margin": "N/A"}
return fundamental_records

def calculate_momentum_metrics(df_history, tickers):
"""Quantitatively screens data for volume velocity and explosive breakout flags."""
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

perf_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
recent_vol_avg = volume.iloc[-20:-1].mean()
vol_velocity = volume.iloc[-1] / recent_vol_avg if recent_vol_avg > 0 else 1.0

tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
atr_20 = tr.rolling(20).mean().iloc[-1]
current_tr = tr.iloc[-1]

atr_ratio = current_tr / atr_20 if atr_20 > 0 else 1.0
is_breakout = atr_ratio >= 1.5

rankings.append({
"Ticker": ticker,
"Price": round(close.iloc[-1], 2),
"20D Return (%)": round(perf_20d, 2),
"Vol Velocity (x)": round(vol_velocity, 2),
"ATR (20)": round(atr_20, 2),
"TR/ATR Ratio": round(atr_ratio, 2),
"Explosive Flag": "🔥 BREAKOUT" if is_breakout else "Normal"
})
except Exception:
continue

df_rank = pd.DataFrame(rankings)
if not df_rank.empty:
df_rank['Score'] = df_rank['20D Return (%)'] * df_rank['Vol Velocity (x)']
return df_rank.sort_values(by='Score', ascending=False).head(10).drop(columns=['Score'])
return df_rank

def calculate_sentiment_score(df_history, ticker, lookback=20):
"""Calculates a synthetic Fear & Greed Sentiment Score (0-100) using technical market proxies."""
try:
close = df_history['Close'][ticker].dropna()
high = df_history['High'][ticker].dropna()
low = df_history['Low'][ticker].dropna()

if len(close) < lookback + 1:
raise ValueError("Insufficient data points for rolling calculations.")

delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs.iloc[-1]))
rsi_score = np.nan_to_num(rsi, nan=50.0)

sma_20 = close.rolling(window=20).mean().iloc[-1]
current_price = close.iloc[-1]
price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

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
"volatility_ratio": round(vol_ratio, 2)
}
}
except Exception as e:
return {
"timestamp": datetime.now(timezone.utc).isoformat(),
"ticker": ticker,
"score": 50,
"label": "Neutral (Insufficient Data)",
"error": str(e)
}

def calculate_advanced_sentiment(df_history, ticker):
"""Wrapper for technical sentiment calculation."""
try:
sentiment_result = calculate_sentiment_score(df_history, ticker)
return {
"status": "Active",
"score": sentiment_result.get("score", 50),
"label": sentiment_result.get("label", "Neutral"),
"timestamp": sentiment_result.get("timestamp"),
"metrics": sentiment_result.get("metrics", {}),
"error": sentiment_result.get("error")
}
except Exception as e:
return {"status": "Error", "score": 50, "label": "Error in calculation", "error": str(e)}

def calculate_macro_trends(df_history, tickers, fundamental_data):
"""Evaluates long-term technical market regimes combined with core corporate fundamental parameters."""
macro_data = []
if df_history.empty:
return pd.DataFrame()

for ticker in tickers:
try:
close = df_history['Close'][ticker].dropna()
if len(close) < 200:
continue

sma_50 = close.rolling(window=50).mean().iloc[-1]
sma_200 = close.rolling(window=200).mean().iloc[-1]
current_price = close.iloc[-1]

dist_from_sma200 = ((current_price - sma_200) / sma_200) * 100
perf_6month = ((current_price - close.iloc[-126]) / close.iloc[-126]) * 100 if len(close) >= 126 else 0.0

if current_price > sma_50 and sma_50 > sma_200:
regime = "Bullish Extension 🚀"
elif current_price > sma_200 and sma_50 < sma_200:
regime = "Accumulation Phase ⏳"
elif current_price < sma_200 and sma_50 > sma_200:
regime = "Distribution/Correction ⚠️"
else:
regime = "Bear Market Cycle 📉"

# Pull fundamental cache values cleanly
f_metrics = fundamental_data.get(ticker, {"Market Cap": "N/A", "P/E Ratio": "N/A", "Profit Margin": "N/A"})

macro_data.append({
"Ticker": ticker,
"Current Price": round(current_price, 2),
"Market Cap": f_metrics["Market Cap"],
"P/E Ratio": f_metrics["P/E Ratio"],
"Profit Margin": f_metrics["Profit Margin"],
"Dist. from 200D (%)": round(dist_from_sma200, 2),
"6M Return (%)": round(perf_6month, 2),
"Macro Structure": regime
})
except Exception:
continue
return pd.DataFrame(macro_data)


# --- 4. USER INTERFACE PLATFORM ---
st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()

# Global Parallel Backend Sync
with st.spinner("Syncing technical historical structures..."):
historical_data = fetch_historical_data(universe)

with st.spinner("Extracting corporate fundamental structures..."):
fundamental_cache = fetch_fundamental_metrics(universe)

# Master Layout Tabs
tab_momentum, tab_sentiment, tab_macro = st.tabs([
"⚡ Short-Term Momentum",
"🔮 Technical Sentiment",
"🏛️ Macro Wealth & Long-Term Investment"
])

# TAB 1: SHORT-TERM MOMENTUM
with tab_momentum:
st.subheader("Explosive Short-Term Breakout Scanner")
if not historical_data.empty:
momentum_df = calculate_momentum_metrics(historical_data, universe)
st.dataframe(momentum_df, use_container_width=True, hide_index=True)
else:
st.error("Failed to load short-term historical metrics.")

# TAB 2: TECHNICAL SENTIMENT
with tab_sentiment:
st.subheader("Dynamic Fear & Greed Structural Proxies")
selected_ticker = st.selectbox("Select Target Engine Asset:", universe)

if not historical_data.empty:
sentiment = calculate_advanced_sentiment(historical_data, selected_ticker)
if sentiment["status"] == "Active":
col1, col2, col3 = st.columns(3)
with col1:
st.metric("Aggregate Score", sentiment["score"], sentiment["label"])
with col2:
st.metric("RSI (14 Daily)", sentiment["metrics"]["rsi_14"])
with col3:
st.metric("Volatility Multiplier", f"{sentiment['metrics']['volatility_ratio']}x")
else:
st.error(f"Engine Fault: {sentiment['error']}")

# TAB 3: INTEGRATED LONG-TERM INVESTMENT MODALITY (TECHNICAL + FUNDAMENTAL)
with tab_macro:
st.subheader("Institutional Macro Structural & Fundamental Scanner")
st.markdown("This module cross-references technical moving averages with corporate value parameters for high-conviction allocation.")

if not historical_data.empty:
macro_df = calculate_macro_trends(historical_data, universe, fundamental_cache)

if not macro_df.empty:
# Actionable filtering layout matrix
f_col1, f_col2 = st.columns(2)
with f_col1:
regimes = ["All"] + list(macro_df["Macro Structure"].unique())
selected_regime = st.selectbox("Filter Portfolio Regime Structure:", regimes)
with f_col2:
# Add fundamental filtering option
pe_filter = st.radio("Valuation Sorting Priority:", ["None", "Lowest P/E First", "Highest Margin First"])

filtered_df = macro_df if selected_regime == "All" else macro_df[macro_df["Macro Structure"] == selected_regime]

# Apply dynamic frontend pandas table sorting structures based on choice
if pe_filter == "Lowest P/E First":
# Convert 'N/A' strings temporarily to infinity to push them down the rank cleanly
filtered_df = filtered_df.assign(pe_numeric=pd.to_numeric(filtered_df['P/E Ratio'], errors='coerce').fillna(np.inf))
filtered_df = filtered_df.sort_values(by="pe_numeric", ascending=True).drop(columns=['pe_numeric'])
elif pe_filter == "Highest Margin First":
filtered_df = filtered_df.assign(margin_numeric=pd.to_numeric(filtered_df['Profit Margin'].str.replace('%',''), errors='coerce').fillna(-np.inf))
filtered_df = filtered_df.sort_values(by="margin_numeric", ascending=False).drop(columns=['margin_numeric'])
else:
filtered_df = filtered_df.sort_values(by="Dist. from 200D (%)", ascending=True)

st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# Interactive visualization context for investment entries
st.subheader("Macro Trend Construction Visualization")
viz_ticker = st.selectbox("Select Asset for Multi-Month Visual Inspection:", filtered_df["Ticker"].tolist() if not filtered_df.empty else universe)

try:
ticker_close = historical_data["Close"][viz_ticker].dropna()
t_50 = ticker_close.rolling(50).mean()
t_200 = ticker_close.rolling(200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=ticker_close.index, y=ticker_close, name="Spot Price", line=dict(color="#38bdf8")))
fig.add_trace(go.Scatter(x=t_50.index, y=t_50, name="50D SMA (Cyclical Trend)", line=dict(color="#f59e0b", dash="dash")))
fig.add_trace(go.Scatter(x=t_200.index, y=t_200, name="200D SMA (Institutional Base)", line=dict(color="#ef4444", width=2)))

fig.update_layout(title=f"{viz_ticker} Structural Health Matrix", template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)
except Exception as e:
st.caption(f"Could not build visualization matrix for {viz_ticker}: {e}")
else:
st.warning("Insufficient structural pricing matrix to process 200-day horizons.")
else:
st.error("Engine Fault: Macro framework history inaccessible.")
