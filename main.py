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
        "MRAM", "ASTS", "ANET", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ",
        "RKLB", "SNDK", "CYBR", "INTC", "F", "PLTR", "SOUN", "BBAI", "NOW", "CIFR",
        "AVGO", "MU", "STX", "LITE"
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

# --- SENTIMENT VISUALIZATION ENGINE ---
st.markdown("### Sentiment Structure Visualization")

try:
    ticker_df = historical_data[selected_ticker].dropna()
    close = ticker_df["Close"]
    high = ticker_df["High"]
    low = ticker_df["Low"]

    # --- RSI 14 ---
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=rsi_series.index,
        y=rsi_series,
        mode="lines",
        name="RSI 14",
        line=dict(color="#38bdf8", width=2)
    ))
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.15, line_width=0)
    fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.15, line_width=0)
    fig_rsi.update_layout(
        title=f"{selected_ticker} — RSI (14)",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # --- PRICE vs SMA20 ---
    sma20 = close.rolling(20).mean()

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=close.index,
        y=close,
        name="Close",
        line=dict(color="#38bdf8", width=2)
    ))
    fig_price.add_trace(go.Scatter(
        x=sma20.index,
        y=sma20,
        name="SMA20",
        line=dict(color="#f59e0b", dash="dash")
    ))
    fig_price.update_layout(
        title=f"{selected_ticker} — Price vs SMA20",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # --- VOLATILITY RATIO MINI-CHART ---
    tr = np.maximum(
        (high - low),
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )
    atr5 = tr.rolling(5).mean()
    atr20 = tr.rolling(20).mean()
    vol_ratio_series = atr5 / atr20

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=vol_ratio_series.index,
        y=vol_ratio_series,
        name="ATR5 / ATR20",
        line=dict(color="#ef4444", width=2)
    ))
    fig_vol.update_layout(
        title=f"{selected_ticker} — Volatility Ratio",
        template="plotly_dark",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # --- DISPLAY ---
    st.plotly_chart(fig_price, use_container_width=True)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.plotly_chart(fig_vol, use_container_width=True)

except Exception as e:
    st.error(f"Visualization Engine Fault: {e}")


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

# TAB 2: TECHNICAL SENTIMENT (PREMIUM RIBBON ENGINE)
with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")
    selected_ticker = st.selectbox("Select Target Engine Asset:", universe)

    if not historical_data.empty:

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

            # --- CALCULATE RSI ---
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))

            # --- SMA20 ---
            sma20 = close.rolling(20).mean()

            # --- VOLATILITY RATIO ---
            tr = np.maximum((high - low),
                            np.maximum(abs(high - close.shift(1)),
                                       abs(low - close.shift(1))))
            atr5 = tr.rolling(5).mean()
            atr20 = tr.rolling(20).mean()
            vol_ratio_series = atr5 / atr20

            # --- PREMIUM TREND RIBBON ENGINE (LUXALGO STYLE) ---
            trend_scores = []      # 0–100
            trend_zones = []       # labels
            trend_colors = []      # rgba colors

            # helper to clamp
            def clamp(x, lo, hi):
                return max(lo, min(hi, x))

            for i in range(len(close)):
                if i < 20:
                    # not enough structure yet
                    trend_scores.append(50)
                    trend_zones.append("Neutral")
                    trend_colors.append("rgba(148, 163, 184, 0.10)")  # slate
                    continue

                price = close.iloc[i]
                ma = sma20.iloc[i]
                rsi_val = rsi_series.iloc[i]
                vol_val = vol_ratio_series.iloc[i]

                # --- COMPONENT 1: Trend vs SMA20 ---
                # +1 if above, -1 if below, scaled by distance
                if pd.isna(ma) or ma == 0:
                    trend_component = 0
                else:
                    dist_pct = (price - ma) / ma * 100
                    trend_component = clamp(dist_pct / 5.0, -1.5, 1.5)

                # --- COMPONENT 2: RSI Bias ---
                # RSI > 55 bullish, < 45 bearish
                rsi_component = clamp((rsi_val - 50) / 15.0, -1.5, 1.5)

                # --- COMPONENT 3: Volatility Regime ---
                # vol_ratio > 1 bullish, < 1 bearish
                vol_component = clamp((vol_val - 1.0) / 0.5, -1.5, 1.5)

                # Weighted composite
                raw_score = (
                    0.45 * trend_component +
                    0.35 * rsi_component +
                    0.20 * vol_component
                )

                # Map to 0–100
                score_0_100 = clamp((raw_score + 1.5) / 3.0 * 100, 0, 100)
                trend_scores.append(score_0_100)

                # Zone classification
                if score_0_100 >= 75:
                    zone = "Strong Bull"
                    color = "rgba(34, 197, 94, 0.18)"   # bright green
                elif score_0_100 >= 60:
                    zone = "Bull"
                    color = "rgba(74, 222, 128, 0.14)"  # soft green
                elif score_0_100 >= 40:
                    zone = "Neutral"
                    color = "rgba(148, 163, 184, 0.10)" # slate
                elif score_0_100 >= 25:
                    zone = "Bear"
                    color = "rgba(248, 113, 113, 0.14)" # soft red
                else:
                    zone = "Strong Bear"
                    color = "rgba(239, 68, 68, 0.20)"   # deep red

                trend_zones.append(zone)
                trend_colors.append(color)

            # --- PRICE CHART WITH PREMIUM RIBBON + SIGNALS ---
            fig_price = go.Figure()

            # Trend ribbon as vertical bands
            for i in range(1, len(close)):
                fig_price.add_vrect(
                    x0=close.index[i-1],
                    x1=close.index[i],
                    fillcolor=trend_colors[i],
                    opacity=1.0,
                    line_width=0,
                    layer="below"
                )

            # Price + SMA20
            fig_price.add_trace(go.Scatter(
                x=close.index, y=close,
                name="Close", line=dict(color="#38bdf8", width=2)
            ))
            fig_price.add_trace(go.Scatter(
                x=sma20.index, y=sma20,
                name="SMA20", line=dict(color="#f59e0b", dash="dash")
            ))

            # --- SIGNAL ENGINE (aligned with ribbon) ---
            buy_signals = []
            sell_signals = []

            for i in range(1, len(close)):
                price = close.iloc[i]
                prev_price = close.iloc[i-1]
                ma = sma20.iloc[i]
                prev_ma = sma20.iloc[i-1]
                rsi_val = rsi_series.iloc[i]
                vol_val = vol_ratio_series.iloc[i]
                score_val = trend_scores[i]

                # Cross conditions
                is_cross_up = (
                    price > ma and
                    prev_price <= prev_ma
                )
                is_cross_down = (
                    price < ma and
                    prev_price >= prev_ma
                )

                is_momentum_strong = rsi_val > 55
                is_momentum_weak = rsi_val < 45
                is_vol_expanding = vol_val > 1.0
                is_vol_contracting = vol_val < 0.8

                # BUY: strong ribbon + bullish cross + momentum + vol
                if (
                    is_cross_up and
                    is_momentum_strong and
                    is_vol_expanding and
                    score_val >= 60
                ):
                    buy_signals.append((close.index[i], price))

                # SELL: ribbon weakening OR bearish cross / weak momentum / vol crush
                if (
                    is_cross_down or
                    is_momentum_weak or
                    is_vol_contracting or
                    score_val <= 40
                ):
                    sell_signals.append((close.index[i], price))

            # BUY markers
            if buy_signals:
                fig_price.add_trace(go.Scatter(
                    x=[t for t, p in buy_signals],
                    y=[p for t, p in buy_signals],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#22c55e", line=dict(color="#16a34a", width=1)),
                    name="BUY"
                ))

            # SELL markers
            if sell_signals:
                fig_price.add_trace(go.Scatter(
                    x=[t for t, p in sell_signals],
                    y=[p for t, p in sell_signals],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(color="#b91c1c", width=1)),
                    name="SELL"
                ))

            fig_price.update_layout(
                title=f"{selected_ticker} — Premium Trend Ribbon & Signals",
                template="plotly_dark",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # --- BACKTEST ENGINE (USING SAME LOGIC) ---
            st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

            returns = []
            position = None
            entry_price = None

            for i in range(1, len(close)):
                price = close.iloc[i]
                ma = sma20.iloc[i]
                rsi_val = rsi_series.iloc[i]
                vol_val = vol_ratio_series.iloc[i]
                score_val = trend_scores[i]

                # ENTRY
                if (
                    position is None and
                    price > ma and
                    rsi_val > 55 and
                    vol_val > 1.0 and
                    score_val >= 60
                ):
                    position = "LONG"
                    entry_price = price

                # EXIT
                elif (
                    position == "LONG" and (
                        price < ma or
                        rsi_val < 45 or
                        vol_val < 0.8 or
                        score_val <= 40
                    )
                ):
                    returns.append((price - entry_price) / entry_price)
                    position = None
                    entry_price = None

            if returns:
                avg_return = np.mean(returns) * 100
                win_rate = (np.sum(np.array(returns) > 0) / len(returns)) * 100
                st.metric("Avg Trade Return", f"{avg_return:.2f}%")
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Total Trades", len(returns))
            else:
                st.info("Not enough signals to compute backtest.")

            # --- PROBABILITY MODEL (TIED TO RIBBON) ---
            st.markdown("### 🔮 Trend Continuation Probability")

            prob = (
                0.45 * (sentiment["metrics"]["rsi_14"] / 100) +
                0.35 * (max(0, sentiment["metrics"]["ma_deviation_pct"]) / 20) +
                0.20 * min(1.5, sentiment["metrics"]["volatility_ratio"]) / 1.5
            )

            probability = min(100, max(0, prob * 100))

            # Blend with latest trend score for a LuxAlgo-style feel
            latest_trend_score = trend_scores[-1] if trend_scores else 50
            blended_prob = 0.6 * probability + 0.4 * latest_trend_score

            st.metric("Continuation Probability", f"{blended_prob:.1f}%")

        else:
            st.error(f"Engine Fault: {sentiment['error']}")

# TAB 3
with tab_macro:
    st.subheader("Institutional Macro Structural & Fundamental Scanner")
    st.markdown("This module cross-references technical moving averages with corporate value parameters.")

    if not historical_data.empty:
        macro_df = calculate_macro_trends(historical_data, universe, fundamental_cache)

        if not macro_df.empty:
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
                )
                filtered_df = filtered_df.sort_values(by="pe_numeric", ascending=True).drop(columns=["pe_numeric"])

            elif pe_filter == "Highest Margin First":
                filtered_df = filtered_df.assign(
                    margin_numeric=pd.to_numeric(filtered_df["Profit Margin"].str.replace("%", ""), errors="coerce").fillna(-np.inf)
                )
                filtered_df = filtered_df.sort_values(by="margin_numeric", ascending=False).drop(columns=["margin_numeric"])

            else:
                filtered_df = filtered_df.sort_values(by="Dist. from 200D (%)", ascending=True)

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            st.subheader("Macro Trend Construction Visualization")
            viz_ticker = st.selectbox(
                "Select Asset for Multi-Month Visual Inspection:",
                filtered_df["Ticker"].tolist() if not filtered_df.empty else universe
            )

            try:
                available_tickers = historical_data.columns.get_level_values(0).unique()
                if viz_ticker in available_tickers:
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
                else:
                    st.caption(f"No pricing structures found for {viz_ticker}")
            except Exception as e:
                st.caption(f"Could not build visualization for {viz_ticker}: {e}")

        else:
            st.warning("Insufficient structural pricing matrix for 200-day horizons.")
    else:
        st.error("Engine Fault: Macro framework history inaccessible.")
