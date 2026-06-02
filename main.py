import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

# Custom CSS
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
    return [
        "ASTS", "ANET", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "ARM", "ZS", "APP", "NASA", "DPRO", "UMAC",
        "RKLB", "SNDK", "CYBR", "INTC", "CIFR", "RDDT", "QUBT", "QBTS", "SNOW", "HIVE", "ONDS", "F",
        "AVGO", "MU", "STX", "QCOM", "TE", "BE", "APLD", "CLSK", "CRWV", "KEEL", "CORZ", "ONDS", "IREN", "NBIS",
        "ENPH", "QCOM", "SMCI", "RGTI", "ASTC", "SHOP", "FJET", "NVDA", "SHAZ", "WOLF", "AVAV", "RCAT", "KTOS", "BA",
    ]

@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=730):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, group_by="ticker", progress=False)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_fundamental_metrics(tickers):
    fundamental_records = {}
    for ticker in tickers:
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info

            raw_cap = info.get("marketCap", None)
            if raw_cap and raw_cap >= 1e12:
                cap_str = f"${raw_cap / 1e12:.2f}T"
            elif raw_cap and raw_cap >= 1e9:
                cap_str = f"${raw_cap / 1e9:.2f}B"
            elif raw_cap and raw_cap >= 1e6:
                cap_str = f"${raw_cap / 1e6:.2f}M"
            else:
                cap_str = "N/A"

            margin_raw = info.get("profitMargins", None)
            margin_pct = f"{margin_raw * 100:.2f}%" if margin_raw is not None else "N/A"

            fundamental_records[ticker] = {
                "Market Cap": cap_str,
                "P/E Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else "N/A",
                "Profit Margin": margin_pct
            }
        except Exception:
            fundamental_records[ticker] = {
                "Market Cap": "N/A",
                "P/E Ratio": "N/A",
                "Profit Margin": "N/A"
            }
    return fundamental_records

def calculate_momentum_metrics(df_history, tickers):
    rankings = []
    if df_history.empty:
        return pd.DataFrame()

    available_tickers = df_history.columns.get_level_values(0).unique()

    for ticker in tickers:
        try:
            if ticker not in available_tickers:
                continue

            ticker_df = df_history[ticker].dropna()
            if len(ticker_df) < 20:
                continue

            close = ticker_df['Close']
            volume = ticker_df['Volume']
            high = ticker_df['High']
            low = ticker_df['Low']

            perf_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            recent_vol_avg = volume.iloc[-20:-1].mean()
            vol_velocity = volume.iloc[-1] / recent_vol_avg if recent_vol_avg > 0 else 1.0

            tr = np.maximum(
                (high - low),
                np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
            )
            atr_20 = tr.rolling(20).mean().iloc[-1]

            sig = unified_signal(ticker_df)
            structure = classify_structure(sig)

            rankings.append({
                "Ticker": ticker,
                "Price": round(close.iloc[-1], 2),
                "20D Return (%)": round(perf_20d, 2),
                "Vol Velocity (x)": round(vol_velocity, 2),
                "ATR (20)": round(atr_20, 2),
                "TR/ATR Ratio": round(tr.iloc[-1] / atr_20 if atr_20 > 0 else 1.0, 2),
                "Explosive Flag": structure
            })
        except Exception:
            continue

    df_rank = pd.DataFrame(rankings)
    if not df_rank.empty:
        df_rank["Score"] = df_rank["20D Return (%)"] * df_rank["Vol Velocity (x)"]
        return df_rank.sort_values(by="Score", ascending=False).head(10).drop(columns=["Score"])

    return df_rank

def calculate_sentiment_score(df_history, ticker, lookback=20):
    try:
        available_tickers = df_history.columns.get_level_values(0).unique()
        if ticker not in available_tickers:
            raise ValueError(f"Ticker {ticker} not found.")

        ticker_df = df_history[ticker].dropna()
        close = ticker_df["Close"]
        high = ticker_df["High"]
        low = ticker_df["Low"]

        if len(close) < lookback + 1:
            raise ValueError("Insufficient data.")

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        rsi_score = np.nan_to_num(rsi, nan=50.0)

        sma_20 = close.rolling(20).mean().iloc[-1]
        current_price = close.iloc[-1]
        price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
        ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )
        atr_5 = tr.rolling(5).mean().iloc[-1]
        atr_20 = tr.rolling(20).mean().iloc[-1]
        vol_ratio = atr_5 / atr_20 if atr_20 > 0 else 1
        vol_score = np.interp(vol_ratio, [0.8, 1.5], [80, 20])

        composite_score = int(np.average(
            [rsi_score, ma_score, vol_score],
            weights=[0.4, 0.4, 0.2]
        ))

        if composite_score >= 75:
            label = "Extreme Greed"
        elif composite_score >= 55:
            label = "Greed"
        elif composite_score >= 45:
            label = "Neutral"
        elif composite_score >= 25:
            label = "Fear"
        else:
            label = "Extreme Fear"

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
        return {"status": "Error", "score": 50, "label": "Error", "error": str(e)}

def calculate_macro_trends(df_history, tickers, fundamental_data):
    macro_data = []
    if df_history.empty:
        return pd.DataFrame()

    available_tickers = df_history.columns.get_level_values(0).unique()

    for ticker in tickers:
        try:
            if ticker not in available_tickers:
                continue

            ticker_df = df_history[ticker].dropna()
            close = ticker_df["Close"]
            if len(close) < 200:
                continue

            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]
            current_price = close.iloc[-1]

            dist_from_sma200 = ((current_price - sma_200) / sma_200) * 100
            perf_6month = (
                (current_price - close.iloc[-126]) / close.iloc[-126]
            ) * 100 if len(close) >= 126 else 0.0

            sig = unified_signal(ticker_df)
            regime = classify_structure(sig)

            f_metrics = fundamental_data.get(ticker, {
                "Market Cap": "N/A",
                "P/E Ratio": "N/A",
                "Profit Margin": "N/A"
            })

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

def compute_factor_scores(df_history, ticker, fundamentals):
    try:
        df = df_history[ticker].dropna()
        close = df["Close"]

        ret_1m = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else 0
        ret_3m = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100 if len(close) > 63 else 0
        ret_6m = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126] * 100 if len(close) > 126 else 0

        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
        trend_strength = (
            3 if close.iloc[-1] > sma50 > sma200 else
            1 if close.iloc[-1] > sma200 else
            -1 if sma50 > sma200 else
            -3
        )

        high = df["High"]
        low = df["Low"]
        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )
        atr20 = tr.rolling(20).mean().iloc[-1]
        volatility = atr20 / close.iloc[-1]
        stability = 1 / volatility if volatility > 0 else 0

        pe = fundamentals.get("P/E Ratio", "N/A")
        margin = fundamentals.get("Profit Margin", "N/A")
        margin_val = float(margin.replace("%", "")) if margin != "N/A" else 0
        pe_val = float(pe) if pe != "N/A" else 50

        quality = margin_val
        value = 1 / pe_val if pe_val > 0 else 0
        growth = ret_6m

        composite = (
            (ret_3m * 0.25) +
            (trend_strength * 10 * 0.25) +
            (stability * 20 * 0.25) +
            (quality * 0.15) +
            (value * 50 * 0.10)
        )

        return {
            "1M": ret_1m,
            "3M": ret_3m,
            "6M": ret_6m,
            "Trend": trend_strength,
            "Volatility": volatility,
            "Stability": stability,
            "Quality": quality,
            "Value": value,
            "Growth": growth,
            "Composite": composite
        }

    except Exception:
        return None

# --- UNIFIED SIGNAL ENGINE + STRUCTURE CLASSIFIER ---
def unified_signal(df):
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    high = df["High"]
    low = df["Low"]
    tr = np.maximum(
        (high - low),
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )
    atr20 = tr.rolling(20).mean()
    atr5 = tr.rolling(5).mean()
    vol_ratio = atr5 / atr20

    return {
        "price": float(close.iloc[-1]),
        "sma20": float(sma20.iloc[-1]),
        "sma50": float(sma50.iloc[-1]),
        "sma200": float(sma200.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "vol_ratio": float(vol_ratio.iloc[-1])
    }

def classify_structure(sig):
    price = sig["price"]
    sma20 = sig["sma20"]
    sma50 = sig["sma50"]
    sma200 = sig["sma200"]
    rsi = sig["rsi"]
    vol = sig["vol_ratio"]

    breakout = (price > sma20 and rsi > 55 and vol > 1.1)
    mid_trend = price > sma50
    long_trend = price > sma200

    if breakout:
        return "Short-Term Breakout 🚀"
    if mid_trend and long_trend:
        return "Healthy Uptrend 📈"
    if long_trend:
        return "Accumulation ⏳"
    return "Neutral / Wait ⚪"

# --- AI ENGINE FOR TAB 4 ---
def build_ai_stock_selection_table(df_history, universe, fundamental_cache):
    rows = []
    if df_history.empty:
        return pd.DataFrame()

    available = df_history.columns.get_level_values(0).unique()

    for ticker in universe:
        if ticker not in available:
            continue

        try:
            df = df_history[ticker].dropna()
            if len(df) < 80:
                continue

            sig = unified_signal(df)
            structure = classify_structure(sig)

            sentiment = calculate_advanced_sentiment(df_history, ticker)
            sent_score = sentiment.get("score", 50)

            fundamentals = fundamental_cache.get(ticker, {
                "Market Cap": "N/A",
                "P/E Ratio": "N/A",
                "Profit Margin": "N/A"
            })
            factor = compute_factor_scores(df_history, ticker, fundamentals)
            if factor is None:
                continue

            ai_score = (
                sent_score * 0.35 +
                factor["3M"] * 0.25 +
                factor["Stability"] * 10 * 0.20 +
                factor["Quality"] * 0.10 +
                factor["Value"] * 100 * 0.10
            )
            ai_score = float(np.clip(ai_score, 0, 100))

            rows.append({
                "Ticker": ticker,
                "Price": round(df["Close"].iloc[-1], 2),
                "AI Score": round(ai_score, 1),
                "Sentiment Score": sent_score,
                "3M Return (%)": round(factor["3M"], 2),
                "Stability": round(factor["Stability"], 2),
                "Quality": round(factor["Quality"], 2),
                "Value": round(factor["Value"], 4),
                "Structure": structure,
                "Market Cap": fundamentals["Market Cap"],
                "P/E Ratio": fundamentals["P/E Ratio"],
                "Profit Margin": fundamentals["Profit Margin"],
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="AI Score", ascending=False).reset_index(drop=True)

# --- 4. USER INTERFACE PLATFORM ---
st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()

st.sidebar.markdown("### ➕ Add Custom Stocks")

# Manual text input (comma-separated)
manual_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):",
    placeholder="e.g., TSLA, AAPL, PLTR"
)

# Convert to list
manual_list = []
if manual_input:
    manual_list = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

# Multi-select dropdown
custom_select = st.sidebar.multiselect(
    "Or select from universe:",
    options=universe,
    default=[]
)

# Merge all tickers
user_added_tickers = list(set(manual_list + custom_select))

# Final universe = base + user-added
full_universe = list(set(universe + user_added_tickers))

st.sidebar.success(f"Tracking {len(full_universe)} total tickers")

with st.spinner("Syncing technical historical structures..."):
    historical_data = fetch_historical_data(full_universe)

with st.spinner("Extracting corporate fundamental structures..."):
    fundamental_cache = fetch_fundamental_metrics(full_universe)

# Precompute AI table once so it's available for Top 3 panel
ai_df = pd.DataFrame()
if not historical_data.empty:
    ai_df = build_ai_stock_selection_table(historical_data, full_universe, fundamental_cache)

tab_momentum, tab_sentiment, tab_macro, tab_ai = st.tabs([
    "⚡ Short-Term Momentum",
    "🔮 Technical Sentiment",
    "🏛️ Macro Wealth & Long-Term Investment",
    "🤖 AI Stock Selection Engine"
])

# --- TAB 1: SHORT-TERM MOMENTUM ---
with tab_momentum:
    st.subheader("Explosive Short-Term Breakout Scanner")
    if not historical_data.empty:
        momentum_df = calculate_momentum_metrics(historical_data, full_universe)
        if not momentum_df.empty:
            st.dataframe(momentum_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No assets matched momentum lookup thresholds.")
    else:
        st.error("Failed to load short-term historical metrics.")

# --- TAB 2: TECHNICAL SENTIMENT ---
def compute_signal_quality_and_narrative(
    close,
    sma20,
    rsi_series,
    vol_ratio_series,
    returns,
    win_rate,
    avg_return,
    trend_phase
):
    latest_price = float(close.iloc[-1])
    latest_sma20 = float(sma20.iloc[-1])
    latest_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    latest_vol = float(vol_ratio_series.iloc[-1]) if not np.isnan(vol_ratio_series.iloc[-1]) else 1.0

    if latest_sma20 > 0:
        dist_sma20_pct = (latest_price - latest_sma20) / latest_sma20 * 100
    else:
        dist_sma20_pct = 0.0

    trend_score = float(np.interp(
        dist_sma20_pct,
        [-5, 0, 5, 10],
        [0, 40, 70, 100]
    ))

    momentum_score = float(np.interp(
        latest_rsi,
        [30, 45, 55, 70],
        [0, 40, 70, 100]
    ))

    vol_score = float(np.interp(
        latest_vol,
        [0.6, 0.9, 1.1, 1.6],
        [20, 80, 60, 20]
    ))

    if returns:
        win_component = float(np.interp(
            win_rate,
            [30, 50, 70],
            [20, 60, 100]
        ))
        ret_component = float(np.interp(
            avg_return,
            [-2, 0, 2],
            [20, 60, 100]
        ))
        backtest_score = win_component * 0.6 + ret_component * 0.4
    else:
        backtest_score = 50.0

    if trend_phase == "Short-Term Breakout 🚀":
        structure_score = 100.0
    elif trend_phase == "Healthy Uptrend 📈":
        structure_score = 80.0
    elif trend_phase == "Accumulation ⏳":
        structure_score = 60.0
    else:
        structure_score = 40.0

    signal_quality = (
        trend_score * 0.30 +
        momentum_score * 0.25 +
        vol_score * 0.20 +
        backtest_score * 0.15 +
        structure_score * 0.10
    )
    signal_quality = round(float(signal_quality), 1)

    narrative_lines = []

    if trend_score >= 75:
        narrative_lines.append("Price is advancing above its short-term trend base with strong directional alignment.")
    elif trend_score >= 50:
        narrative_lines.append("Price is hovering near its short-term trend base, with a developing directional bias.")
    else:
        narrative_lines.append("Price is trading below key short-term trend levels, signaling caution.")

    if momentum_score >= 75:
        narrative_lines.append("RSI reflects firm bullish momentum with strong buying pressure.")
    elif momentum_score >= 50:
        narrative_lines.append("Momentum is balanced, with neither buyers nor sellers in clear control.")
    else:
        narrative_lines.append("RSI indicates fading momentum and a weaker demand profile.")

    if latest_vol > 1.2:
        narrative_lines.append("Volatility is expanding, increasing the probability of sharp swings and breakout-type moves.")
    elif latest_vol < 0.9:
        narrative_lines.append("Volatility is compressed, often preceding future expansion phases.")
    else:
        narrative_lines.append("Volatility is operating within a normal regime for this asset.")

    if returns:
        if win_rate > 60 and avg_return > 0:
            narrative_lines.append("Historical signals show a favorable skew with a positive average trade outcome.")
        elif win_rate > 50:
            narrative_lines.append("Historical signals show a modest positive edge, but with mixed outcomes.")
        else:
            narrative_lines.append("Historical signals do not yet demonstrate a strong or persistent edge.")
    else:
        narrative_lines.append("Insufficient historical signal data to characterize backtested trade outcomes.")

    if trend_phase == "Short-Term Breakout 🚀":
        narrative_lines.append("Structural regime aligns with a short-term breakout phase, favoring momentum continuation setups.")
    elif trend_phase == "Healthy Uptrend 📈":
        narrative_lines.append("Structural regime confirms a healthy multi-timeframe uptrend, supportive of trend-following strategies.")
    elif trend_phase == "Accumulation ⏳":
        narrative_lines.append("Structural regime suggests accumulation behavior, often preceding more decisive trend moves.")
    else:
        narrative_lines.append("Structural regime is neutral, with no strong directional bias confirmed.")

    return signal_quality, narrative_lines, {
        "trend_score": round(trend_score, 1),
        "momentum_score": round(momentum_score, 1),
        "vol_score": round(vol_score, 1),
        "backtest_score": round(backtest_score, 1),
        "structure_score": round(structure_score, 1),
    }

with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")
    selected_ticker = st.selectbox("Select Target Engine Asset:", full_universe)

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

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment["score"],
                title={"text": "Sentiment Gauge (0–100)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 25], "color": "#1e3a8a"},
                        {"range": [25, 45], "color": "#0f766e"},
                        {"range": [45, 55], "color": "#475569"},
                        {"range": [55, 75], "color": "#ca8a04"},
                        {"range": [75, 100], "color": "#b91c1c"},
                    ],
                }
            ))
            st.plotly_chart(gauge_fig, use_container_width=True)

            ticker_df = historical_data[selected_ticker].dropna()
            close = ticker_df["Close"]
            high = ticker_df["High"]
            low = ticker_df["Low"]

            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            sma20 = close.rolling(20).mean()

            tr = np.maximum(
                (high - low),
                np.maximum(abs(high - close.shift(1)),
                           abs(low - close.shift(1)))
            )
            atr5 = tr.rolling(5).mean()
            atr20 = tr.rolling(20).mean()
            vol_ratio_series = atr5 / atr20

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=close.index, y=close,
                name="Close", line=dict(color="#38bdf8", width=2)
            ))
            fig_price.add_trace(go.Scatter(
                x=sma20.index, y=sma20,
                name="SMA20", line=dict(color="#f59e0b", dash="dash")
            ))

            buy_signals = []
            sell_signals = []

            for i in range(1, len(close)):
                if (
                    close.iloc[i] > sma20.iloc[i] and
                    close.iloc[i-1] <= sma20.iloc[i-1] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    buy_signals.append((close.index[i], close.iloc[i]))

                if (
                    (close.iloc[i] < sma20.iloc[i] and
                     close.iloc[i-1] >= sma20.iloc[i-1]) or
                    rsi_series.iloc[i] < 45
                ):
                    sell_signals.append((close.index[i], close.iloc[i]))

            for t, p in buy_signals:
                fig_price.add_annotation(
                    x=t, y=p, text="⬆ BUY",
                    showarrow=True, arrowhead=1,
                    font=dict(color="#22c55e")
                )

            for t, p in sell_signals:
                fig_price.add_annotation(
                    x=t, y=p, text="⬇ SELL",
                    showarrow=True, arrowhead=1,
                    font=dict(color="#ef4444")
                )

            fig_price.update_layout(
                title=f"{selected_ticker} — Price with Signals",
                template="plotly_dark", height=320
            )
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("### Sentiment Structure Visualization")

            try:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=rsi_series.index, y=rsi_series,
                    mode="lines", name="RSI 14",
                    line=dict(color="#38bdf8", width=2)
                ))
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.15, line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.15, line_width=0)
                fig_rsi.update_layout(
                    title=f"{selected_ticker} — RSI (14)",
                    template="plotly_dark", height=230
                )

                fig_price2 = go.Figure()
                fig_price2.add_trace(go.Scatter(
                    x=close.index, y=close,
                    name="Close", line=dict(color="#38bdf8", width=2)
                ))
                fig_price2.add_trace(go.Scatter(
                    x=sma20.index, y=sma20,
                    name="SMA20", line=dict(color="#f59e0b", dash="dash")
                ))
                fig_price2.update_layout(
                    title=f"{selected_ticker} — Price vs SMA20",
                    template="plotly_dark", height=260
                )

                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=vol_ratio_series.index, y=vol_ratio_series,
                    name="ATR5 / ATR20",
                    line=dict(color="#ef4444", width=2)
                ))
                fig_vol.update_layout(
                    title=f"{selected_ticker} — Volatility Ratio",
                    template="plotly_dark", height=230
                )

                st.plotly_chart(fig_price2, use_container_width=True)
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.plotly_chart(fig_vol, use_container_width=True)

            except Exception as e:
                st.error(f"Visualization Engine Fault: {e}")

            st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

            returns = []
            trade_lengths = []
            position = None
            entry_price = None
            entry_index = None

            for i in range(1, len(close)):
                if (
                    position is None and
                    close.iloc[i] > sma20.iloc[i] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    position = "LONG"
                    entry_price = close.iloc[i]
                    entry_index = i

                elif (
                    position == "LONG" and (
                        close.iloc[i] < sma20.iloc[i] or
                        rsi_series.iloc[i] < 45 or
                        vol_ratio_series.iloc[i] < 0.8
                    )
                ):
                    ret = (close.iloc[i] - entry_price) / entry_price
                    returns.append(ret)
                    if entry_index is not None:
                        trade_lengths.append(i - entry_index)
                    position = None
                    entry_price = None
                    entry_index = None

            avg_return = np.mean(returns) * 100 if returns else 0.0
            win_rate = (np.sum(np.array(returns) > 0) / len(returns)) * 100 if returns else 0.0
            avg_length = np.mean(trade_lengths) if trade_lengths else 0
            median_length = np.median(trade_lengths) if trade_lengths else 0

            if returns:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Avg Trade Return", f"{avg_return:.2f}%")
                with c2:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                with c3:
                    st.metric("Avg Hold (Days)", f"{avg_length:.1f}")
                with c4:
                    st.metric("Median Hold (Days)", f"{median_length:.1f}")
            else:
                st.info("Not enough signals to compute backtest.")

            st.markdown("### 📊 Pattern-Based Growth Projection")

            if returns:
                last_n = min(10, len(returns))
                proj_10d = np.mean(returns[-last_n:]) * 100
                proj_20d = np.mean(returns) * 100
                confidence = win_rate

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Projected 10D Trend", f"{proj_10d:.2f}%")
                with c2:
                    st.metric("Projected 20D Trend", f"{proj_20d:.2f}%")
                with c3:
                    st.metric("Pattern Confidence", f"{confidence:.1f}%")
            else:
                st.caption("Projection unavailable: insufficient historical signal data.")

            st.markdown("### 🧭 Trend Phase & Scenario Map")

            sig = unified_signal(ticker_df)
            trend_phase = classify_structure(sig)

            cont_prob = (
                0.4 * (sentiment["metrics"]["rsi_14"] / 100) +
                0.4 * (max(0, sentiment["metrics"]["ma_deviation_pct"]) / 20) +
                0.2 * min(1.5, sentiment["metrics"]["volatility_ratio"]) / 1.5
            )
            cont_prob = float(min(1, max(0, cont_prob)))
            latest_vol = float(vol_ratio_series.iloc[-1]) if not np.isnan(vol_ratio_series.iloc[-1]) else 1.0
            pullback_prob = float(min(0.6, max(0, latest_vol - 1)))
            sideway_prob = float(max(0, 1 - cont_prob - pullback_prob))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Trend Phase", trend_phase)
            with c2:
                st.metric("Continuation Scenario", f"{cont_prob*100:.1f}%")
            with c3:
                st.metric("Sideways Scenario", f"{sideway_prob*100:.1f}%")
            with c4:
                st.metric("Pullback Scenario", f"{pullback_prob*100:.1f}%")
        else:
            st.error("Sentiment engine returned error state.")
    else:
        st.error("Historical data unavailable.")

# --- TAB 3: MACRO WEALTH & LONG-TERM ---
with tab_macro:
    st.subheader("Macro Wealth & Long-Term Investment Structure")

    if historical_data.empty:
        st.error("Historical data unavailable.")
    else:
        macro_df = calculate_macro_trends(historical_data, universe, fundamental_cache)
        if macro_df.empty:
            st.warning("No macro structures could be derived from current dataset.")
        else:
            st.dataframe(macro_df, use_container_width=True, hide_index=True)

# --- TAB 4: AI STOCK SELECTION ENGINE ---
with tab_ai:
    st.subheader("🤖 AI Stock Selection Engine")

    if historical_data.empty:
        st.error("Historical data unavailable.")
    else:
        if ai_df.empty:
            with st.spinner("Running AI multi-factor engine..."):
                local_ai_df = build_ai_stock_selection_table(historical_data, full_universe, fundamental_cache)
        else:
            local_ai_df = ai_df

        if local_ai_df.empty:
            st.warning("No assets passed AI engine filters.")
        else:
            st.dataframe(local_ai_df, use_container_width=True, hide_index=True)

            top = local_ai_df.iloc[0]
            st.markdown("### 🏆 Top AI Pick")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Ticker", top["Ticker"])
            with c2:
                st.metric("AI Score", top["AI Score"])
            with c3:
                st.metric("Sentiment Score", top["Sentiment Score"])

            st.markdown("## 🏆 Top 3 AI Picks")

            top3 = local_ai_df.head(3)

            card_style = """
                <style>
                .ai-card {
                    background: rgba(15, 23, 42, 0.55);
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    border-radius: 12px;
                    padding: 18px;
                    margin-bottom: 12px;
                    backdrop-filter: blur(12px);
                }
                .ai-rank {
                    font-size: 22px;
                    font-weight: 700;
                    color: #38bdf8;
                }
                .ai-ticker {
                    font-size: 28px;
                    font-weight: 800;
                    color: #f8fafc;
                }
                .ai-score {
                    font-size: 22px;
                    font-weight: 700;
                    color: #22c55e;
                }
                .ai-structure {
                    font-size: 16px;
                    color: #cbd5e1;
                }
                </style>
            """
            st.markdown(card_style, unsafe_allow_html=True)

            for idx, row in top3.iterrows():
                rank_label = ["🥇 #1", "🥈 #2", "🥉 #3"][idx]

                st.markdown(f"""
                    <div class="ai-card">
                        <div class="ai-rank">{rank_label}</div>
                        <div class="ai-ticker">{row['Ticker']}</div>
                        <div class="ai-score">AI Score: {row['AI Score']}</div>
                        <div class="ai-structure">{row['Structure']}</div>
                        <br>
                        <div style="color:#94a3b8;">
                            Sentiment: {row['Sentiment Score']} • 
                            3M Return: {row['3M Return (%)']}% • 
                            Stability: {row['Stability']} • 
                            Quality: {row['Quality']} • 
                            Value: {row['Value']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
