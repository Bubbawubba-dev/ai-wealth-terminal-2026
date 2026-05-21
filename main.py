import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Wealth Terminal v13.0",
    layout="wide",
    page_icon="📈"
)

# --- GLOBAL UI THEME (v13.0) ---
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Background */
body {
    background: radial-gradient(circle at top, #020617, #020617);
}

/* Glass cards */
.glass-card {
    background: rgba(15,23,42,0.85);
    border-radius: 18px;
    padding: 18px 20px;
    border: 1px solid rgba(148,163,184,0.25);
    backdrop-filter: blur(14px);
}

/* Neon KPI */
.neon-kpi {
    font-size: 30px;
    font-weight: 700;
    color: #38bdf8;
    text-shadow: 0 0 14px rgba(56,189,248,0.9);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(15,23,42,0.9);
    border-radius: 12px 12px 0 0;
    padding: 10px 20px;
    border: 1px solid rgba(51,65,85,0.8);
    transition: 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(30,64,175,0.6);
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.18);
    border-bottom: 2px solid #38bdf8;
    color: #e5e7eb !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    border: none;
    padding: 0.5rem 1.1rem;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 0.9rem;
    transition: 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-1px) scale(1.03);
    box-shadow: 0 0 18px rgba(56,189,248,0.7);
}

/* Metric cards */
.metric-card {
    background: rgba(15,23,42,0.9);
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px solid rgba(51,65,85,0.9);
}

/* Dataframe tweaks */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Subtle divider */
.hr-glow {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    margin: 0.8rem 0 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# --- 2. SECURITY + SIDEBAR ---

def check_password():
    if "password_correct" not in st.session_state:
        with st.sidebar:
            st.markdown("### 🔐 Access")
            pwd = st.text_input("Access Key", type="password")
            if st.button("Unlock"):
                if pwd == st.secrets.get("APP_PASSWORD", "1234"):
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid key")
        return False
    return True

if not check_password():
    st.stop()

with st.sidebar:
    st.markdown("## 📈 Wealth Terminal")
    st.caption("v13.0 · Sleek Quant Surface")

    st.markdown("<hr class='hr-glow'>", unsafe_allow_html=True)

    st.markdown("#### Universe Mode")
    universe_mode = st.radio(
        "Universe",
        ["Core Tech Momentum", "Full Base Universe"],
        label_visibility="collapsed"
    )

    # you can later branch on universe_mode if you want
    st.markdown("<hr class='hr-glow'>", unsafe_allow_html=True)

    st.markdown("#### Display Options")
    show_sentiment_tab = st.checkbox("Show Technical Sentiment", value=True)
    show_macro_tab = st.checkbox("Show Macro & Fundamentals", value=True)

    st.markdown("<hr class='hr-glow'>", unsafe_allow_html=True)
    st.caption("Built for 10–30 day swing structures.")


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

universe = get_base_universe()

with st.spinner("Syncing technical historical structures..."):
    historical_data = fetch_historical_data(universe)

with st.spinner("Extracting corporate fundamental structures..."):
    fundamental_cache = fetch_fundamental_metrics(universe)

# --- HERO HEADER ---
st.markdown("""
<div class='glass-card' style='margin-bottom:12px;'>
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='font-size:14px; color:#9ca3af;'>Wealth Terminal v13.0</div>
            <div class='neon-kpi'>Institutional Quant Surface</div>
            <div style='font-size:13px; color:#64748b; margin-top:4px;'>
                Momentum · Sentiment · Macro · Fundamentals
            </div>
        </div>
        <div style='text-align:right; font-size:12px; color:#6b7280;'>
            Session Timezone<br>
            <span style='color:#e5e7eb;'>Asia / Hong Kong</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- COMPACT TOP STRIP ---
st.markdown(f"""
<div style="
    width:100%;
    padding:10px 18px;
    margin-bottom:14px;
    border-radius:14px;
    background:rgba(15,23,42,0.75);
    backdrop-filter:blur(12px);
    border:1px solid rgba(148,163,184,0.18);
    display:flex;
    justify-content:space-between;
    align-items:center;
    box-shadow:0 0 18px rgba(56,189,248,0.15);
">

    <!-- LEFT -->
    <div style="display:flex; gap:22px; align-items:center;">
        <div style="color:#38bdf8; font-weight:600; font-size:15px;">
            Terminal: <span style="color:#e2e8f0;">Online</span>
        </div>

        <div style="color:#38bdf8; font-weight:600; font-size:15px;">
            Universe: <span style="color:#e2e8f0;">{len(universe)} assets</span>
        </div>

        <div style="color:#38bdf8; font-weight:600; font-size:15px;">
            Sync: <span style="color:#e2e8f0;">Live</span>
        </div>
    </div>

    <!-- RIGHT -->
    <div style="text-align:right; color:#94a3b8; font-size:13px;">
        HK Time<br>
        <span style="color:#e2e8f0;">{datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%H:%M:%S")}</span>
    </div>

</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab_momentum, tab_sentiment, tab_macro = st.tabs([
    " Momentum",
    " Technical Sentiment",
    " Macro & Long-Term"
])

# TAB 1
with tab_momentum:
    st.markdown("## <span style='color:#38bdf8;'> Short‑Term Momentum Scanner</span>", unsafe_allow_html=True)
    if not historical_data.empty:
        momentum_df = calculate_momentum_metrics(historical_data, universe)
        if not momentum_df.empty:
            st.dataframe(momentum_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No assets matched momentum lookup thresholds.")
    else:
        st.error("Failed to load short-term historical metrics.")

# TAB 2: TECHNICAL SENTIMENT
with tab_sentiment:
    st.markdown("## <span style='color:#f472b6;'> Technical Sentiment Engine</span>", unsafe_allow_html=True)
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

            # --- SIGNAL ARROWS ---
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
                    close.iloc[i] < sma20.iloc[i] and
                    close.iloc[i-1] >= sma20.iloc[i-1] or
                    rsi_series.iloc[i] < 45
                ):
                    sell_signals.append((close.index[i], close.iloc[i]))

            # Plot arrows
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
                template="plotly_dark", height=300
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # --- BACKTEST ENGINE ---
            st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

            returns = []
            position = None
            entry_price = None

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

                # EXIT
                elif (
                    position == "LONG" and (
                        close.iloc[i] < sma20.iloc[i] or
                        rsi_series.iloc[i] < 45 or
                        vol_ratio_series.iloc[i] < 0.8
                    )
                ):
                    returns.append((close.iloc[i] - entry_price) / entry_price)
                    position = None
                    entry_price = None

            # Summary
            if returns:
                avg_return = np.mean(returns) * 100
                win_rate = (np.sum(np.array(returns) > 0) / len(returns)) * 100
                st.metric("Avg Trade Return", f"{avg_return:.2f}%")
                st.metric("Win Rate", f"{win_rate:.1f}%")
                st.metric("Total Trades", len(returns))
            else:
                st.info("Not enough signals to compute backtest.")

            # --- PROBABILITY MODEL ---
            st.markdown("### 🔮 Trend Continuation Probability")

            prob = (
                0.4 * (sentiment["metrics"]["rsi_14"] / 100) +
                0.4 * (max(0, sentiment["metrics"]["ma_deviation_pct"]) / 20) +
                0.2 * min(1.5, sentiment["metrics"]["volatility_ratio"]) / 1.5
            )

            probability = min(100, max(0, prob * 100))

            st.metric("Continuation Probability", f"{probability:.1f}%")

        else:
            st.error(f"Engine Fault: {sentiment['error']}")

# TAB 3
with tab_macro:
    st.markdown("## <span style='color:#a78bfa;'>🏛️ Macro Wealth Framework</span>", unsafe_allow_html=True)
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

            st.markdown("## <span style='color:#a78bfa;'> Macro Wealth Framework</span>", unsafe_allow_html=True)

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
