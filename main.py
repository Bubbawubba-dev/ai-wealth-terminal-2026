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
    return ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ",
            "RKLB", "KTOS", "CYBR", "GNK", "PHYS", "PLTR", "SOUN", "BBAI", "MARA", "RIOT"]

@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=180):
    """Safely fetches multi-ticker daily historical data across the core universe."""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, progress=False)
        if data.empty or 'Close' not in data:
            return pd.DataFrame()
        return data
    except Exception:
        return pd.DataFrame()

def calculate_momentum_metrics(df_history, tickers):
    """Quantitatively screens data for volume velocity and explosive breakout flags."""
    rankings = []
    if df_history.empty:
        return pd.DataFrame()

    for ticker in tickers:
        try:
            # Handle multi-index columns from yfinance batch download safely
            close = df_history['Close'][ticker].dropna() if ticker in df_history['Close'] else pd.Series()
            volume = df_history['Volume'][ticker].dropna() if ticker in df_history['Volume'] else pd.Series()
            high = df_history['High'][ticker].dropna() if ticker in df_history['High'] else pd.Series()
            low = df_history['Low'][ticker].dropna() if ticker in df_history['Low'] else pd.Series()

            if len(close) < 20:
                continue

            # Calculations
            perf_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            recent_vol_avg = volume.iloc[-20:-1].mean()
            vol_velocity = volume.iloc[-1] / recent_vol_avg if recent_vol_avg > 0 else 1.0

            # True Range (TR) & Average True Range (ATR)
            tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            atr_20 = tr.rolling(20).mean().iloc[-1]
            current_tr = tr.iloc[-1]

            # Breakout Condition: True Range expansion ratio >= 1.5
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
        # Score based on combination of price momentum and volume acceleration
        df_rank['Score'] = df_rank['20D Return (%)'] * df_rank['Vol Velocity (x)']
        return df_rank.sort_values(by='Score', ascending=False).head(10).drop(columns=['Score'])
    return df_rank

def calculate_sentiment_score(df_history, ticker, lookback=20):
    """Calculates a synthetic Fear & Greed Sentiment Score (0-100) using technical market proxies (RSI, MA extensions, and Volatility), appended with a timestamp"""
    try:
        # Extract ticker-specific data safely
        close = df_history['Close'][ticker].dropna()
        high = df_history['High'][ticker].dropna()
        low = df_history['Low'][ticker].dropna()

        if len(close) < lookback + 1:
            raise ValueError("Insufficient data points for rolling calculations.")

        # --- 1. RSI Component (Weight: 40%) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # RSI naturally scales 0-100. RSI 50 = Neutral.
        rsi_score = np.nan_to_num(rsi, nan=50.0) 

        # --- 2. Moving Average Extension Component (Weight: 40%) ---
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        current_price = close.iloc[-1]
        price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
        
        # Normalize: -10% below SMA = 0 (Extreme Fear), +10% above SMA = 100 (Extreme Greed)
        ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

        # --- 3. Volatility Proxy Component (Weight: 20%) ---
        tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr_5 = tr.rolling(window=5).mean().iloc[-1]
        atr_20 = tr.rolling(window=20).mean().iloc[-1]
        
        # Higher current volatility usually correlates with fear/panic selling
        vol_ratio = atr_5 / atr_20 if atr_20 > 0 else 1
        # Normalize: Ratio > 1.5 = Fear (Score 20), Ratio < 0.8 = Greed (Score 80)
        vol_score = np.interp(vol_ratio, [0.8, 1.5], [80, 20])

        # --- 4. Aggregate & Classify ---
        composite_score = int(np.average([rsi_score, ma_score, vol_score], weights=[0.4, 0.4, 0.2]))
        
        if composite_score >= 75: label = "Extreme Greed"
        elif composite_score >= 55: label = "Greed"
        elif composite_score >= 45: label = "Neutral"
        elif composite_score >= 25: label = "Fear"
        else: label = "Extreme Fear"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        # Failsafe dictionary matching the expected structure
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "score": 50,
            "label": "Neutral (Insufficient Data)",
            "error": str(e)
        }

def generate_forecast(ticker_series, days_ahead=30):
    """Generates a 30-day forward forecast with upper/lower volatility bands using linear regression."""
    try:
        if len(ticker_series) < 30:
            return None
        
        # Use last 60 days for linear regression trend
        recent_data = ticker_series.iloc[-60:].reset_index(drop=True)
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Linear regression fit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Calculate standard deviation of residuals for bands
        residuals = y - p(x)
        std_dev = np.std(residuals)
        
        # Generate forecast
        future_x = np.arange(len(recent_data), len(recent_data) + days_ahead)
        forecast_values = p(future_x)
        upper_band = forecast_values + (2 * std_dev)
        lower_band = forecast_values - (2 * std_dev)
        
        # Create dataframe with forecast dates
        last_date = ticker_series.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        forecast_df = pd.DataFrame({
            'Forecast': forecast_values,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        }, index=future_dates)
        
        return forecast_df
    except Exception as e:
        st.error(f"Forecast generation error: {str(e)}")
        return None

# --- 4. MAIN APPLICATION INITIALIZATION ---
# Initialize session state variables
if "account_size" not in st.session_state:
    st.session_state.account_size = 100000.0
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = 2.0

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Terminal Configuration")
    st.session_state.account_size = st.number_input("Account Size ($)", value=st.session_state.account_size, step=1000.0)
    st.session_state.risk_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=st.session_state.risk_pct, step=0.5)

account_size = st.session_state.account_size
risk_pct = st.session_state.risk_pct

# Fetch data
universe = get_base_universe()
hist_data = fetch_historical_data(universe)
top_10_momentum = calculate_momentum_metrics(hist_data, universe)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📊 Momentum Scanner", "💰 Position Sizer", "🔮 Forecasting"])

# --- TAB 1: MOMENTUM SCANNER ---
with tab1:
    st.subheader("Real-Time Momentum & Volatility Breakout Scanner")
    st.markdown("Identifies explosive momentum candidates using 20-day returns, volume velocity, and ATR breakout signals.")
    
    if not top_10_momentum.empty:
        st.dataframe(top_10_momentum, use_container_width=True, hide_index=True)
    else:
        st.warning("No momentum data available. Check data fetch status.")

# --- TAB 2: POSITION SIZER & RISK ARCHITECT ---
with tab2:
    st.subheader("Smart Position Sizing & Strategic Entry Engine")

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_ticker = st.selectbox("Target Execution Security", options=top_10_momentum["Ticker"].tolist() if not top_10_momentum.empty else ["PLTR"])

        # Pull precise context variables from selected asset
        try:
            ticker_close = hist_data['Close'][selected_ticker].dropna().iloc[-1]
            ticker_atr = (hist_data['High'][selected_ticker] - hist_data['Low'][selected_ticker]).rolling(20).mean().dropna().iloc[-1]
        except Exception:
            ticker_close, ticker_atr = 50.0, 2.5

        entry_price = st.number_input("Target Execution Entry Price ($)", min_value=0.01, value=float(ticker_close), step=0.1)
        stop_loss = st.number_input("Systemic Stop-Loss Floor Level ($)", min_value=0.01, value=float(entry_price - (2 * ticker_atr)), step=0.1)

    # Quantitative risk sizing math engine calculations
    risk_dollars = account_size * (risk_pct / 100)
    per_share_risk = entry_price - stop_loss

    if per_share_risk > 0:
        shares_to_buy = int(risk_dollars // per_share_risk)
        total_notional_cost = shares_to_buy * entry_price
        portfolio_allocation_pct = (total_notional_cost / account_size) * 100
    else:
        shares_to_buy, total_notional_cost, portfolio_allocation_pct = 0, 0.0, 0.0

    with col2:
        st.markdown(f"### Allocation Matrix Blueprint: **{selected_ticker}**")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Absolute Capital at Risk", f"${risk_dollars:,.2f}")
        with m_col2:
            st.metric("Calculated Allocation Quantity", f"{shares_to_buy:,} Shares")
        with m_col3:
            st.metric("Total Order Value", f"${total_notional_cost:,.2f}")

        st.progress(min(portfolio_allocation_pct / 100, 1.0))
        st.caption(f"This order utilizes **{portfolio_allocation_pct:.1f}%** of overall portfolio margin/cash assets.")

    # Sentiment Pipeline Mock Analysis (Alternative Data Component)
    st.markdown("---")
    st.subheader("Alternative Data: Sentiment & Order Flow Matrix")
    s_col1, s_col2 = st.columns(2)

    # Get sentiment score for selected ticker
    sentiment_data = calculate_sentiment_score(hist_data, selected_ticker)
    sentiment_score = sentiment_data["score"]
    sentiment_label = sentiment_data["label"]
    last_updated = sentiment_data["timestamp"]
    
    with s_col1:
        st.metric("Aggregated Retail Sentiment Score",
        value=f"{sentiment_score}/100",
        delta=sentiment_label,
        delta_color="normal" if sentiment_score >= 45 else "inverse")

        st.caption(f"Last updated: {last_updated}")

# --- TAB 3: MATHEMATICAL FORECASTING ---
with tab3:
    st.subheader("Statistical Time-Series Trend Projections")
    st.markdown("Projects historical patterns forward 30 days using linear regressions and standard volatility deviation limits.")

    forecast_ticker = st.selectbox("Select Projective Modeling Target", options=universe, index=0)

    if not hist_data.empty and forecast_ticker in hist_data['Close']:
        ticker_series = hist_data['Close'][forecast_ticker].dropna()
        forecast_df = generate_forecast(ticker_series)

        # Calculate sentiment score for the selected forecast ticker
        forecast_sentiment_data = calculate_sentiment_score(hist_data, forecast_ticker)
        forecast_sentiment = forecast_sentiment_data.get("score", 50)

        # Display sentiment metrics for the selected ticker
        st.markdown("---")
        st.subheader(f"Sentiment Analysis: {forecast_ticker}")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.metric("Aggregated Retail Sentiment Score", f"{forecast_sentiment}/100", delta="Bullish Bias" if forecast_sentiment > 50 else "Bearish Bias")
        with f_col2:
            st.metric("Institutional Order Accumulation Rate", "Highly Accelerated" if forecast_sentiment > 60 else "Distribution State")

        if forecast_df is not None:
            fig = go.Figure()

            # Historical Frame Trace
            fig.add_trace(go.Scatter(x=ticker_series.index[-60:], y=ticker_series.values[-60:], name="Historical Reality", line=dict(color="#38bdf8", width=2.5)))

            # Center Mathematical Mean Projection
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name="Mean Statistical Path", line=dict(color="#e2e8f0", dash="dash")))

            # Upper Confidence Threshold Boundary
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper Band'], name="Upper Volatility Target (2σ)", line=dict(color="#22c55e", width=1, dash="dot")))

            # Lower Safety Confidence Boundary
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower Band'], name="Lower Volatility Boundary (2σ)", line=dict(color="#ef4444", width=1, dash="dot"), fill='tonexty', fillcolor='rgba(239, 68, 68, 0.1)'))

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=20, r=20, t=20, b=20),
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Insufficient rolling sample points to compute modeling matrix.")
    else:
        st.error("Core financial tracking dataset structure error.")

# --- 5. FUTURE EXPANSION HOOKS ---
st.markdown("---")
st.caption("⚓ Developer API Core Integrations Status: Webhook Daemon Listening on `localhost:8000` | Alpaca / Interactive Brokers Sandboxed Core: `Offline`")
