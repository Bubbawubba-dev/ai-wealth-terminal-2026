import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo 

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

# Custom CSS to improve terminal UI scannability
st.markdown("""
<style>
.metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 15px;}
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
            "RKLB", "BTDR", "CYBR", "GNK", "F", "PLTR", "SOUN", "BBAI", "NOW", "CIFR", 
            "AVGO", "MU", "STX", "LITE"]

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
        rsi_score = np.nan_to_num(rsi, nan=50.0) 

        # --- 2. Moving Average Extension Component (Weight: 40%) ---
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        current_price = close.iloc[-1]
        price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
        ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

        # --- 3. Volatility Proxy Component (Weight: 20%) ---
        tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr_5 = tr.rolling(window=5).mean().iloc[-1]
        atr_20 = tr.rolling(window=20).mean().iloc[-1]
        
        vol_ratio = atr_5 / atr_20 if atr_20 > 0 else 1
        vol_score = np.interp(vol_ratio, [0.8, 1.5], [80, 20])

        # --- 4. Aggregate & Classify ---
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
    """Wrapper for technical sentiment calculation - returns structured technical sentiment data."""
    sentiment_result = calculate_sentiment_score(df_history, ticker)
    return {
        "status": "Active" if "error" not in sentiment_result else "Error",
        "score": sentiment_result.get("score", 50),
        "label": sentiment_result.get("label", "Neutral"),
        "timestamp": sentiment_result.get("timestamp"),
        "metrics": sentiment_result.get("metrics", {}),
        "error": sentiment_result.get("error", None)
    }

# --- NEW QUANT ENGINE: SYSTEMIC RISK MODULES ---
@st.cache_data(ttl=86400)
def fetch_financial_ratios(ticker):
    """Fetches Balance Sheet leverage parameters (Debt/Equity, Current Ratio) safely."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        de_ratio = info.get("debtToEquity", None)
        curr_ratio = info.get("currentRatio", None)
        return {
            "Ticker": ticker,
            "Debt to Equity": round(de_ratio / 100, 2) if de_ratio else "N/A", # Normalize if returned as pct
            "Current Ratio": round(curr_ratio, 2) if curr_ratio else "N/A"
        }
    except Exception:
        return {"Ticker": ticker, "Debt to Equity": "N/A", "Current Ratio": "N/A"}

def calculate_crowded_trades(df_history, tickers):
    """Identifies institutional tracking abnormalities via extreme volume velocity z-scores."""
    crowded_list = []
    if df_history.empty:
        return pd.DataFrame()
        
    for ticker in tickers:
        try:
            volume = df_history['Volume'][ticker].dropna()
            if len(volume) < 40: continue
            
            recent_vol = volume.iloc[-1]
            hist_mean = volume.iloc[-40:-1].mean()
            hist_std = volume.iloc[-40:-1].std()
            
            z_score = (recent_vol - hist_mean) / hist_std if hist_std > 0 else 0
            
            if z_score > 2.0: flag = "⚠️ CRITICAL OVERCROWDING"
            elif z_score > 1.0: flag = "⚡ Elevated Interest"
            else: flag = "Normal"
            
            crowded_list.append({
                "Ticker": ticker,
                "Current Vol": int(recent_vol),
                "Vol Z-Score": round(z_score, 2),
                "Crowding Risk": flag
            })
        except Exception:
            continue
    return pd.DataFrame(crowded_list).sort_values(by="Vol Z-Score", ascending=False)

def calculate_volatility_skew():
    """Generates a dynamic market skew profile mapping index safe-haven option demands."""
    try:
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]
        vxn = yf.Ticker("^VXN").history(period="1d")["Close"].iloc[-1]
        skew_ratio = vix / vxn if vxn > 0 else 1.0
        
        if skew_ratio > 1.15: label = "⚠️ High Downside Hedging (Broad Market Focus)"
        elif skew_ratio < 0.85: label = "⚠️ Extreme Tech Sector Hedging Risk"
        else: label = "Balanced Asset Class Risk Distribution"
        
        return {"VIX": round(vix, 2), "VXN": round(vxn, 2), "Ratio": round(skew_ratio, 2), "Status": label}
    except Exception:
        return {"VIX": 0.0, "VXN": 0.0, "Ratio": 1.0, "Status": "Data Connection Offline"}


# --- 4. INTERFACE DISPLAY ORCHESTRATION ---
st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()
hist_data = fetch_historical_data(universe)

# Establish Dashboard Navigation Tabs
tab_main, tab_risk = st.tabs(["🚀 Momentum Engine", "⚠️ Systemic Risk & Tail Protection"])

with tab_main:
    st.header("Core Universe Momentum Leaderboard")
    if not hist_data.empty:
        momentum_df = calculate_momentum_metrics(hist_data, universe)
        st.dataframe(momentum_df, use_container_width=True, hide_index=True)
        
        st.subheader("Single Ticker Sentiment Check")
        selected_ticker = st.selectbox("Select Target Ticker", universe)
        sentiment = calculate_advanced_sentiment(hist_data, selected_ticker)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Technical Sentiment Score", sentiment["score"])
        with col2:
            st.metric("Risk Label", sentiment["label"])
        with col3:
            st.metric("Last Evaluated Zone (HK)", str(sentiment["timestamp"]))
    else:
        st.error("Error connecting to primary market data providers.")

with tab_risk:
    st.header("Tail-Risk & Black Swan Mitigation Metrics")
    
    # Row 1: Skew Meter & Corporate Debt Metrics Configuration
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("📊 Cross-Asset Volatility Skew Meter")
        skew_data = calculate_volatility_skew()
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Tail Risk Skew Profile</h4>
            <p><b>Market State:</b> {skew_data['Status']}</p>
            <hr style="border-color:#334155;">
            <p>🔴 <b>S&P 500 VIX:</b> {skew_data['VIX']}</p>
            <p>🔵 <b>Nasdaq 100 VXN:</b> {skew_data['VXN']}</p>
            <p>🎛️ <b>VIX/VXN Skew Ratio:</b> {skew_data['Ratio']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.subheader("🏛️ Corporate Debt & Solvency Ratios")
        target_co = st.selectbox("Analyze Balance Sheet Leverage for Asset:", universe, index=8)
        
        with st.spinner("Parsing SEC Fundamental Filings..."):
            ratios = fetch_financial_ratios(target_co)
            
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"{target_co} Debt-to-Equity (D/E)", value=ratios["Debt to Equity"], 
                      help="Measures relative company leverage. Metrics over 2.0 indicate heightened structural tail-risk.")
        with c2:
            st.metric(label=f"{target_co} Current Ratio", value=ratios["Current Ratio"], 
                      help="Measures short term liquidity cushion. Values below 1.0 indicate operational distress potential.")

    st.markdown("---")
    
    # Row 2: Crowded Trades Tracker & Correlation Space
    col_bot_left, col_bot_right = st.columns(2)
    
    with col_bot_left:
        st.subheader("🔥 Crowded Trades & Volume Volatility Acceleration")
        if not hist_data.empty:
            crowd_df = calculate_crowded_trades(hist_data, universe)
            st.dataframe(crowd_df, use_container_width=True, hide_index=True)
        else:
            st.info("Awaiting Historical Input Parameters")
            
    with col_bot_right:
        st.subheader("🌐 Systemic Multi-Asset Correlation Convergence Matrix")
        if not hist_data.empty and 'Close' in hist_data:
            try:
                # Calculate the Pearson correlation matrix across the active historical dataset
                corr_matrix = hist_data['Close'].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=False,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    labels=dict(color="Correlation Coefficient")
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="#f8fafc"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("ℹ️ When correlations converge uniformly toward 1.0 (Dark Red), diversification fail-safes disintegrate, indicating impending market structural fragility.")
            except Exception as e:
                st.error(f"Correlation Processing Interrupted: {e}")
        else:
            st.info("Historical data matrix format incomplete.")
