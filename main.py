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
        "MRAM", "ASTS", "ANET", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "ARM", 
        "RKLB", "SNDK", "CYBR", "INTC", "CIFR", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "QUBT", "QBTS",
        "AVGO", "MU", "STX", "LITE", "TE", "BE", "APLD", "CLSK", "CRWV", "KEEL", "CORZ", "WYFI", "IREN", "NBIS"
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

            if current_price > sma_50 and sma_50 > sma_200:
                regime = "Bullish Extension 🚀"
            elif current_price > sma_200 and sma_50 < sma_200:
                regime = "Accumulation Phase ⏳"
            elif current_price < sma_200 and sma_50 > sma_200:
                regime = "Distribution/Correction ⚠️"
            else:
                regime = "Bear Market Cycle 📉"

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

        # Momentum factors
        ret_1m = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100 if len(close) > 21 else 0
        ret_3m = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100 if len(close) > 63 else 0
        ret_6m = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126] * 100 if len(close) > 126 else 0

        # Trend factors
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50
        trend_strength = (
            3 if close.iloc[-1] > sma50 > sma200 else
            1 if close.iloc[-1] > sma200 else
            -1 if sma50 > sma200 else
            -3
        )

        # Volatility
        high = df["High"]
        low = df["Low"]
        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )
        atr20 = tr.rolling(20).mean().iloc[-1]
        volatility = atr20 / close.iloc[-1]
        stability = 1 / volatility if volatility > 0 else 0

        # Fundamentals
        pe = fundamentals.get("P/E Ratio", "N/A")
        margin = fundamentals.get("Profit Margin", "N/A")
        margin_val = float(margin.replace("%", "")) if margin != "N/A" else 0
        pe_val = float(pe) if pe != "N/A" else 50

        quality = margin_val
        value = 1 / pe_val if pe_val > 0 else 0
        growth = ret_6m

        # Composite score
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


# --- 4. USER INTERFACE PLATFORM ---
st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()

with st.spinner("Syncing technical historical structures..."):
    historical_data = fetch_historical_data(universe)

with st.spinner("Extracting corporate fundamental structures..."):
    fundamental_cache = fetch_fundamental_metrics(universe)

tab_momentum, tab_sentiment, tab_macro = st.tabs([
    "⚡ Short-Term Momentum",
    "🔮 Technical Sentiment",
    "🏛️ Macro Wealth & Long-Term Investment"
])

# TAB 1
with tab_momentum:
    st.subheader("Explosive Short-Term Breakout Scanner")
    if not historical_data.empty:
        momentum_df = calculate_momentum_metrics(historical_data, universe)
        if not momentum_df.empty:
            st.dataframe(momentum_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No assets matched momentum lookup thresholds.")
    else:
        st.error("Failed to load short-term historical metrics.")

# TAB 2: TECHNICAL SENTIMENT — UPGRADED
with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")
    selected_ticker = st.selectbox("Select Target Engine Asset:", universe)

    if not historical_data.empty:

        # --- TICKER SENTIMENT ---
        sentiment = calculate_advanced_sentiment(historical_data, selected_ticker)

        if sentiment["status"] == "Active":

            # --- METRICS ROW ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Aggregate Score", sentiment["score"], sentiment["label"])
            with col2:
                st.metric("RSI (14 Daily)", sentiment["metrics"]["rsi_14"])
            with col3:
                st.metric("Volatility Multiplier", f"{sentiment['metrics']['volatility_ratio']}x")

            # --- SENTIMENT GAUGE ---
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

            # --- LOAD PRICE DATA ---
            ticker_df = historical_data[selected_ticker].dropna()
            close = ticker_df["Close"]
            high = ticker_df["High"]
            low = ticker_df["Low"]
            volume = ticker_df.get("Volume", None)

            # --- CALCULATE RSI ---
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            # --- SMA20 ---
            sma20 = close.rolling(20).mean()

            # --- VOLATILITY RATIO ---
            tr = np.maximum(
                (high - low),
                np.maximum(abs(high - close.shift(1)),
                           abs(low - close.shift(1)))
            )
            atr5 = tr.rolling(5).mean()
            atr20 = tr.rolling(20).mean()
            vol_ratio_series = atr5 / atr20

            # --- PRICE CHART WITH SIGNAL ARROWS ---
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
                # BUY
                if (
                    close.iloc[i] > sma20.iloc[i] and
                    close.iloc[i-1] <= sma20.iloc[i-1] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    buy_signals.append((close.index[i], close.iloc[i]))

                # SELL
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

            # --- SENTIMENT VISUALIZATION ENGINE ---
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

            # --- BACKTEST ENGINE + HOLD LENGTH ---
            st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

            returns = []
            trade_lengths = []
            position = None
            entry_price = None
            entry_index = None

            for i in range(1, len(close)):

                # ENTRY
                if (
                    position is None and
                    close.iloc[i] > sma20.iloc[i] and
                    rsi_series.iloc[i] > 50 and
                    vol_ratio_series.iloc[i] > 1.0
                ):
                    position = "LONG"
                    entry_price = close.iloc[i]
                    entry_index = i

                # EXIT
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

            if returns:
                avg_return = np.mean(returns) * 100
                win_rate = (np.sum(np.array(returns) > 0) / len(returns)) * 100
                avg_length = np.mean(trade_lengths) if trade_lengths else 0
                median_length = np.median(trade_lengths) if trade_lengths else 0

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

            # --- GROWTH PROJECTION (PATTERN-BASED) ---
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

            # --- TREND PHASE + SCENARIO MAP ---
            st.markdown("### 🧭 Trend Phase & Scenario Map")

            latest_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50
            latest_vol = float(vol_ratio_series.iloc[-1]) if not np.isnan(vol_ratio_series.iloc[-1]) else 1.0
            latest_price = float(close.iloc[-1])
            latest_sma20 = float(sma20.iloc[-1])

            if latest_rsi < 55 and latest_price > latest_sma20:
                trend_phase = "Early Trend"
            elif 55 <= latest_rsi <= 70:
                trend_phase = "Mid Trend"
            elif latest_rsi > 70 and latest_vol < 1.2:
                trend_phase = "Late Trend"
            elif latest_rsi > 70 and latest_vol >= 1.2:
                trend_phase = "Exhaustion Risk"
            else:
                trend_phase = "Indecisive / Transition"

            # Simple scenario probabilities (heuristic, non-advisory)
            cont_prob = (
                0.4 * (sentiment["metrics"]["rsi_14"] / 100) +
                0.4 * (max(0, sentiment["metrics"]["ma_deviation_pct"]) / 20) +
                0.2 * min(1.5, sentiment["metrics"]["volatility_ratio"]) / 1.5
            )
            cont_prob = float(min(1, max(0, cont_prob)))
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

            # --- MARKET SENTIMENT VS TICKER SENTIMENT ---
            st.markdown("### 🌐 Market vs Ticker Sentiment")

            market_scores = []
            for tk in universe:
                if tk in historical_data.columns.get_level_values(0):
                    s = calculate_advanced_sentiment(historical_data, tk)
                    if s.get("status") == "Active":
                        market_scores.append(s.get("score", 50))

            if market_scores:
                market_sentiment = float(np.mean(market_scores))
                rel_sentiment = sentiment["score"] - market_sentiment

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Ticker Sentiment", sentiment["score"])
                with c2:
                    st.metric("Market Sentiment", f"{market_sentiment:.1f}")
                with c3:
                    st.metric("Relative Sentiment", f"{rel_sentiment:+.1f}")
            else:
                st.caption("Market sentiment benchmark unavailable (insufficient data).")

            # --- RELATIVE SENTIMENT HEATMAP ---
st.markdown("### 🔥 Relative Sentiment Heatmap (Ticker vs Market)")

# Build sentiment table for all tickers
sentiment_table = []

for tk in universe:
    try:
        s = calculate_advanced_sentiment(historical_data, tk)
        if s.get("status") == "Active":
            sentiment_table.append({
                "Ticker": tk,
                "Sentiment": s["score"]
            })
    except:
        pass

sentiment_df = pd.DataFrame(sentiment_table)

if not sentiment_df.empty:
    # Compute relative sentiment
    sentiment_df["Relative"] = sentiment_df["Sentiment"] - sentiment_df["Sentiment"].mean()

    # Pivot for heatmap
    heatmap_df = sentiment_df.set_index("Ticker")[["Relative"]]

    # Build heatmap
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap_df["Relative"],
            x=["Relative Sentiment"],
            y=heatmap_df.index,
            colorscale=[
                [0.0, "#7f1d1d"],   # deep red (underperforming)
                [0.5, "#475569"],   # neutral grey
                [1.0, "#14532d"]    # deep green (outperforming)
            ],
            colorbar=dict(title="Rel. Score")
        )
    )

    fig_heat.update_layout(
        height=400,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("Relative sentiment heatmap unavailable — insufficient data.")

      # --- SIGNAL QUALITY & STORYLINE ---
    st.markdown("### 🧠 Signal Quality & Narrative")

    # Simple signal quality heuristic
    quality_components = []

    if latest_price > latest_sma20:
        quality_components.append(1)
    if 50 <= latest_rsi <= 70:
        quality_components.append(1)
    if 0.8 <= latest_vol <= 1.3:
        quality_components.append(1)
    if returns and win_rate > 50:
        quality_components.append(1)

    signal_quality = (sum(quality_components) / 4) * 100 if quality_components else 0

    st.metric("Signal Quality Score", f"{signal_quality:.1f}/100")

     # Narrative
    narrative_lines = []

    if latest_price > latest_sma20:
        narrative_lines.append("Price is holding above its short-term trend base (SMA20).")
    else:
        narrative_lines.append("Price is trading below its short-term trend base (SMA20).")

    if latest_rsi < 45:
        narrative_lines.append("Momentum is weak, with RSI in a lower band.")
    elif 45 <= latest_rsi <= 60:
        narrative_lines.append("Momentum is balanced, with RSI in a neutral-to-positive zone.")
    elif 60 < latest_rsi <= 70:
        narrative_lines.append("Momentum is strong, with RSI in a bullish band.")
    else:
        narrative_lines.append("Momentum is elevated, suggesting a mature or extended move.")

    if latest_vol > 1.2:
        narrative_lines.append("Volatility is elevated, increasing the risk of sharp swings.")
    elif latest_vol < 0.9:
        narrative_lines.append("Volatility is compressed, often preceding expansion phases.")
    else:
        narrative_lines.append("Volatility is within a normal operating range.")

    if returns:
        if win_rate > 55:
            narrative_lines.append("Historical pattern shows a favorable skew of winning trades.")
        else:
            narrative_lines.append("Historical pattern shows mixed outcomes with no strong edge.")
    else:
        narrative_lines.append("Insufficient historical signal data to characterize trade outcomes.")

        st.write("• " + "\n• ".join(narrative_lines))
    else:
        st.error(f"Engine Fault: {sentiment['error']}")


# TAB 3 — LONG‑TERM MACRO + ENTRY/EXIT ENGINE
with tab_macro:
    st.subheader("🏛️ Institutional Macro Structural & Fundamental Scanner")
    st.markdown("This module cross-references technical moving averages with corporate value parameters and long-term structural entry/exit zones.")

    if historical_data.empty:
        st.error("Engine Fault: Macro framework history inaccessible.")
    else:

        # --- BUILD MACRO TABLE ---
        macro_df = calculate_macro_trends(historical_data, universe, fundamental_cache)

        if macro_df.empty:
            st.warning("Insufficient structural pricing matrix for 200-day horizons.")
        else:

            # --- LONG-TERM ENTRY / EXIT LOGIC ---
            def classify_entry_exit(row):
                dist = row["Dist. from 200D (%)"]
                ret_6m = row["6M Return (%)"]
                regime = row["Macro Structure"]

                # ENTRY ZONE (accumulation bias)
                if regime in ["Bullish Extension 🚀", "Accumulation Phase ⏳"] and -12 <= dist <= +5:
                    entry = "🟢 Accumulation Zone (Near 200D Support)"
                elif regime == "Accumulation Phase ⏳" and dist < -12:
                    entry = "🟡 Deep Value Accumulation (High Patience)"
                elif "Bear" in regime:
                    entry = "🔻 Avoid New Long-Term Entries (Bear Structure)"
                else:
                    entry = "⚪ Neutral / Wait for Better Structure"

                # EXIT / TRIM RISK
                if regime == "Bullish Extension 🚀" and dist > 20 and ret_6m > 30:
                    exit_zone = "🟥 Elevated Trim / Rebalance Risk"
                elif dist > 15 and ret_6m > 20:
                    exit_zone = "🟠 Watch for Exhaustion / Tighten Risk"
                elif dist < -15 and "Bear" in regime:
                    exit_zone = "🔻 Capital Preservation Focus"
                else:
                    exit_zone = "🟢 Hold / No Structural Exit Signal"

                # HOLDING BIAS
                if "Accu" in entry:
                    bias = "Long-Term Accumulation Bias"
                elif "Trim" in exit_zone or "Exhaustion" in exit_zone:
                    bias = "Hold / Trim on Strength"
                elif "Bear" in regime:
                    bias = "Defensive / Underweight Bias"
                else:
                    bias = "Core Hold / Monitor"

                return pd.Series({
                    "Best Entry Zone": entry,
                    "Exit / Trim Zone": exit_zone,
                    "Holding Bias": bias
                })

            macro_df[["Best Entry Zone", "Exit / Trim Zone", "Holding Bias"]] = macro_df.apply(
                classify_entry_exit, axis=1
            )

            # --- LONG-TERM ENTRY WATCHLIST ---
            st.markdown("## 🟢 Long-Term Entry Watchlist (Top Accumulation Candidates)")

            entry_candidates = macro_df[
                macro_df["Best Entry Zone"].str.contains("Accu")
            ].sort_values("Dist. from 200D (%)", ascending=True)

            if entry_candidates.empty:
                st.info("No assets currently in long-term accumulation zones.")
            else:
                st.dataframe(
                    entry_candidates[
                        [
                            "Ticker",
                            "Current Price",
                            "Dist. from 200D (%)",
                            "6M Return (%)",
                            "Macro Structure",
                            "Best Entry Zone"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # --- TRIM RISK RADAR (TABULATED) ---
            st.markdown("## 🟥 Trim Risk Radar — Extended / Overstretched Structures")

            trim_risk = macro_df[
                macro_df["Exit / Trim Zone"].str.contains("Trim|Exhaustion")
            ].copy()

            if trim_risk.empty:
                st.info("No assets currently showing structural trim/exhaustion risk.")
            else:
                trim_risk["Abs Dist from 200D"] = trim_risk["Dist. from 200D (%)"].abs()
                trim_risk["Risk Rank"] = (
                    trim_risk["Abs Dist from 200D"] * 0.6 +
                    trim_risk["6M Return (%)"] * 0.4
                )

                trim_risk = trim_risk.sort_values("Risk Rank", ascending=False)

                st.dataframe(
                    trim_risk[
                        [
                            "Ticker",
                            "Current Price",
                            "Dist. from 200D (%)",
                            "6M Return (%)",
                            "Macro Structure",
                            "Exit / Trim Zone",
                            "Risk Rank"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # --- FILTERS ---
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                regimes = ["All"] + list(macro_df["Macro Structure"].unique())
                selected_regime = st.selectbox("Filter Portfolio Regime Structure:", regimes)
            with f_col2:
                pe_filter = st.radio("Valuation Sorting Priority:", ["None", "Lowest P/E First", "Highest Margin First"])

            filtered_df = macro_df if selected_regime == "All" else macro_df[macro_df["Macro Structure"] == selected_regime]

            if pe_filter == "Lowest P/E First":
                filtered_df = filtered_df.assign(
                    pe_numeric=pd.to_numeric(filtered_df["P/E Ratio"], errors="coerce").fillna(np.inf)
                ).sort_values("pe_numeric").drop(columns=["pe_numeric"])

            elif pe_filter == "Highest Margin First":
                filtered_df = filtered_df.assign(
                    margin_numeric=pd.to_numeric(filtered_df["Profit Margin"].str.replace("%", ""), errors="coerce").fillna(-np.inf)
                ).sort_values("margin_numeric", ascending=False).drop(columns=["margin_numeric"])

            else:
                filtered_df = filtered_df.sort_values("Dist. from 200D (%)")

            # --- DISPLAY TABLE ---
            st.dataframe(
                filtered_df[
                    [
                        "Ticker",
                        "Current Price",
                        "Market Cap",
                        "P/E Ratio",
                        "Profit Margin",
                        "Dist. from 200D (%)",
                        "6M Return (%)",
                        "Macro Structure",
                        "Best Entry Zone",
                        "Exit / Trim Zone",
                        "Holding Bias",
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            # --- VISUALIZATION ---
            st.subheader("Macro Trend Construction Visualization")
            viz_ticker = st.selectbox(
                "Select Asset for Multi-Month Visual Inspection:",
                filtered_df["Ticker"].tolist()
            )

            try:
                ticker_close = historical_data[viz_ticker]["Close"].dropna()
                t_50 = ticker_close.rolling(50).mean()
                t_200 = ticker_close.rolling(200).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ticker_close.index, y=ticker_close, name="Spot Price", line=dict(color="#38bdf8")))
                fig.add_trace(go.Scatter(x=t_50.index, y=t_50, name="50D SMA", line=dict(color="#f59e0b", dash="dash")))
                fig.add_trace(go.Scatter(x=t_200.index, y=t_200, name="200D SMA", line=dict(color="#ef4444", width=2)))

                fig.update_layout(
                    title=f"{viz_ticker} Structural Health Matrix",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.caption(f"Could not build visualization for {viz_ticker}: {e}")



                        
