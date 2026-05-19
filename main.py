import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. CONFIGURATION & EXTERNAL APIS ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

st.markdown("""
<style>
.metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# Initialize NLP Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize Reddit API (Replace with your actual keys from Reddit Developer portal)
try:
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="WealthTerminal_Sentiment_Scraper_v1.0"
    )
    REDDIT_ENABLED = True
except Exception:
    REDDIT_ENABLED = False


# --- 2. BACKEND & DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_base_universe():
    return ["MRAM", "ASTS", "HIMS", "QUBT", "BZFD", "HUT", "FLEX", "VCYT", "VECO", "IONQ",
            "RKLB", "KTOS", "CYBR", "GNK", "PHYS", "PLTR", "SOUN", "BBAI", "MARA", "RIOT"]

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

def generate_forecast(series, periods=30):
    y = series.dropna().values
    x = np.arange(len(y))
    if len(y) < 10:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(y), len(y) + periods)
    forecast_base = slope * future_x + intercept

    resid_std = np.std(y - (slope * x + intercept))
    upper_band = forecast_base + (2 * resid_std)
    lower_band = forecast_base - (2 * resid_std)

    future_dates = [series.index[-1] + timedelta(days=i) for i in range(1, periods + 1)]
    return pd.DataFrame({
        "Forecast": forecast_base,
        "Upper Band": upper_band,
        "Lower Band": lower_band
    }, index=future_dates)

# --- 3. DUAL-STREAM SENTIMENT ENGINES ---
def calculate_advanced_sentiment(df_history, ticker, lookback=20):
    try:
        close = df_history['Close'][ticker].dropna()
        high = df_history['High'][ticker].dropna()
        low = df_history['Low'][ticker].dropna()

        if len(close) < lookback + 1:
            raise ValueError("Insufficient data")

        # 1. RSI Component
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        rsi_score = np.nan_to_num(rsi, nan=50.0)

        # 2. MA Extension Component
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        price_to_sma_pct = ((close.iloc[-1] - sma_20) / sma_20) * 100
        ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

        # 3. Volatility Component
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
            "score": composite_score, "label": label,
            "metrics": {"rsi_14": round(rsi_score, 1), "ma_deviation_pct": round(price_to_sma_pct, 2), "volatility_ratio": round(vol_ratio, 2)}
        }
    except Exception as e:
        return {"score": 50, "label": "Neutral (Error)", "metrics": {}}

@st.cache_data(ttl=900)
def fetch_reddit_nlp_sentiment(ticker, limit=25):
    if not REDDIT_ENABLED or reddit.client_id == "YOUR_CLIENT_ID":
        return {"score": 50, "mentions": 0, "status": "API Keys Required"}

    subreddits = "wallstreetbets+stocks+investing"
    query = f"${ticker} OR {ticker}"
    compound_scores = []
    
    try:
        for submission in reddit.subreddit(subreddits).search(query, sort='new', time_filter='week', limit=limit):
            title_sent = analyzer.polarity_scores(submission.title)['compound']
            body_sent = analyzer.polarity_scores(submission.selftext)['compound'] if submission.selftext else 0
            post_score = (title_sent * 0.7) + (body_sent * 0.3)
            if post_score != 0:
                compound_scores.append(post_score)

        if not compound_scores:
            return {"score": 50, "mentions": 0, "status": "No recent data"}

        avg_compound = np.mean(compound_scores)
        mapped_score = np.interp(avg_compound, [-0.5, 0.5], [0, 100])
        
        return {"score": int(mapped_score), "mentions": len(compound_scores), "status": "Active"}
    except Exception as e:
        return {"score": 50, "mentions": 0, "status": f"Error: {str(e)}"}

def get_dual_sentiment_matrix(df_history, ticker):
    tech_data = calculate_advanced_sentiment(df_history, ticker)
    social_data = fetch_reddit_nlp_sentiment(ticker)
    
    if social_data["status"] == "Active" and social_data["mentions"] > 0:
        s_score = social_data["score"]
        if s_score >= 75: s_label = "Extreme Greed"
        elif s_score >= 55: s_label = "Greed"
        elif s_score >= 45: s_label = "Neutral"
        elif s_score >= 25: s_label = "Fear"
        else: s_label = "Extreme Fear"
    else:
        s_label = "Inactive / No Data"
        
    return {
        "technical": tech_data,
        "social": {"score": social_data["score"], "label": s_label, "mentions": social_data["mentions"], "status": social_data["status"]}
    }


# --- 4. FRONTEND UI & WORKFLOWS ---
st.title("🎛️ Institutional Wealth Terminal")
st.caption("Quantitative Screening, Risk Optimization Analytics, & Advanced Time-Series Projections")

st.sidebar.header("Global Operational Parameters")
account_size = st.sidebar.number_input("Total Portfolio Equity Capital ($)", min_value=1000, value=50000, step=5000)
risk_pct = st.sidebar.slider("Maximum Account Risk Exposure Per Trade (%)", 0.1, 5.0, 1.0, 0.1)

universe = get_base_universe()
hist_data = fetch_historical_data(universe)
top_10_momentum = calculate_momentum_metrics(hist_data, universe)

tab1, tab2, tab3 = st.tabs(["🚀 Momentum Engine", "🛡️ Advanced Risk Architect", "🔮 Mathematical Forecasting"])

# --- TAB 1: MOMENTUM & VOLATILITY SCANNER ---
with tab1:
    st.subheader("Quantitative Scanned Momentum Leaderboard")
    st.markdown("Real-time sorting analyzing compounding **20-day returns** alongside **volume acceleration metrics**.")

    if not top_10_momentum.empty:
        st.dataframe(
            top_10_momentum.style.highlight_max(subset=["Vol Velocity (x)"], color="#1e3a8a")
            .highlight_between(subset=["TR/ATR Ratio"], left=1.5, right=10.0, color="#7f1d1d"),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("Database pipeline error: Historical structural nodes unretrievable.")

# --- TAB 2: POSITION SIZER & RISK ARCHITECT ---
with tab2:
    st.subheader("Smart Position Sizing & Strategic Entry Engine")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_ticker = st.selectbox("Target Execution Security", options=top_10_momentum["Ticker"].tolist() if not top_10_momentum.empty else ["PLTR"])
        
        try:
            ticker_close = hist_data['Close'][selected_ticker].dropna().iloc[-1]
            ticker_atr = (hist_data['High'][selected_ticker] - hist_data['Low'][selected_ticker]).rolling(20).mean().dropna().iloc[-1]
        except Exception:
            ticker_close, ticker_atr = 50.0, 2.5

        entry_price = st.number_input("Target Execution Entry Price ($)", min_value=0.01, value=float(ticker_close), step=0.1)
        stop_loss = st.number_input("Systemic Stop-Loss Floor Level ($)", min_value=0.01, value=float(entry_price - (2 * ticker_atr)), step=0.1)

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
        with m_col1: st.metric("Absolute Capital at Risk", f"${risk_dollars:,.2f}")
        with m_col2: st.metric("Calculated Allocation Quantity", f"{shares_to_buy:,} Shares")
        with m_col3: st.metric("Total Order Value", f"${total_notional_cost:,.2f}")

        st.progress(min(portfolio_allocation_pct / 100, 1.0))
        st.caption(f"This order utilizes **{portfolio_allocation_pct:.1f}%** of overall portfolio margin/cash assets.")

    # --- DUAL SENTIMENT MATRIX & DIVERGENCE ALERT ---
    st.markdown("---")
    st.subheader("Dual-Stream Sentiment Analysis Matrix")
    st.caption("Tracking quantitative price momentum vs. social NLP chatter.")

    sentiment_matrix = get_dual_sentiment_matrix(hist_data, selected_ticker)
    tech = sentiment_matrix["technical"]
    soc = sentiment_matrix["social"]

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("### 📊 Financial Price Sentiment")
        st.metric(
            label="Quantitative Greed/Fear Index", 
            value=f"{tech['score']}/100", 
            delta=tech['label'],
            delta_color="normal" if tech['score'] >= 45 else "inverse",
            help="Derived from 14-period RSI, 20-day MA extensions, and Volatility ratios."
        )
        with st.expander("View Financial Component Breakdown"):
            st.write(f"**RSI (14):** {tech['metrics'].get('rsi_14', 'N/A')}")
            st.write(f"**MA Deviation:** {tech['metrics'].get('ma_deviation_pct', 'N/A')}%")
            st.write(f"**Volatility Ratio:** {tech['metrics'].get('volatility_ratio', 'N/A')}x")

    with s_col2:
        st.markdown("### 🗣️ Alternative Data Sentiment")
        if soc["status"] == "Active" and soc["mentions"] > 0:
            st.metric(
                label="Reddit NLP Sentiment Index", 
                value=f"{soc['score']}/100", 
                delta=soc['label'],
                delta_color="normal" if soc['score'] >= 45 else "inverse",
                help="VADER natural language processing of top financial subreddits."
            )
            st.caption(f"Based on **{soc['mentions']}** recent active discussions.")
        else:
            st.metric(label="Reddit NLP Sentiment Index", value="N/A", delta=soc['status'], delta_color="off")
            st.caption("Waiting for active PRAW connection or more social mentions.")

    # Divergence Alert Check
    if soc["status"] == "Active" and soc["mentions"] > 0:
        score_spread = soc['score'] - tech['score']
        if abs(score_spread) >= 40:
            st.markdown("---")
            if score_spread > 0:
                st.error(
                    "🚨 **DIVERGENCE DETECTED: The 'Bag-Holder' Risk**\n\n"
                    f"**Spread: +{abs(score_spread)} Points (Social > Financial)**\n\n"
                    "Social chatter is aggressively bullish, but quantitative price momentum and volume are weak. "
                    "Retail traders may be buying into a downtrend or institutional distribution phase. Proceed with caution."
                )
            else:
                st.success(
                    "🟢 **DIVERGENCE DETECTED: The 'Smart Money' Setup**\n\n"
                    f"**Spread: +{abs(score_spread)} Points (Financial > Social)**\n\n"
                    "Quantitative momentum is strongly bullish, but retail social sentiment is fearful or entirely "
                    "ignoring this asset. Institutional capital may be quietly driving the trend before retail catches on."
                )


# --- TAB 3: MATHEMATICAL FORECASTING ---
with tab3:
    st.subheader("Statistical Time-Series Trend Projections")
    st.markdown("Projects historical patterns forward 30 days using linear regressions and standard volatility deviation limits.")

    forecast_ticker = st.selectbox("Select Projective Modeling Target", options=universe, index=0)

    if not hist_data.empty and forecast_ticker in hist_data['Close']:
        ticker_series = hist_data['Close'][forecast_ticker].dropna()
        forecast_df = generate_forecast(ticker_series)

        # Utilize the new Dual Matrix for the forecast tab as well
        st.markdown("---")
        st.subheader(f"Sentiment Environment: {forecast_ticker}")
        
        f_matrix = get_dual_sentiment_matrix(hist_data, forecast_ticker)
        f_tech = f_matrix["technical"]
        f_soc = f_matrix["social"]
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.metric("Financial Momentum Proxy", f"{f_tech['score']}/100", delta=f_tech['label'], delta_color="normal" if f_tech['score'] >= 45 else "inverse")
        with f_col2:
            st.metric("Social NLP Chatter Score", f"{f_soc['score']}/100" if f_soc['status'] == "Active" and f_soc['mentions'] > 0 else "N/A", delta=f_soc['label'], delta_color="normal" if f_soc.get('score', 0) >= 45 else "inverse")

        if forecast_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ticker_series.index[-60:], y=ticker_series.values[-60:], name="Historical Reality", line=dict(color="#38bdf8", width=2.5)))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], name="Mean Statistical Path", line=dict(color="#e2e8f0", dash="dash")))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper Band'], name="Upper Volatility Target (2σ)", line=dict(color="#22c55e", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower Band'], name="Lower Volatility Boundary (2σ)", line=dict(color="#ef4444", width=1, dash="dot"), fill='tonexty', fillcolor='rgba(239, 68, 68, 0.2)'))

            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Insufficient rolling sample points to compute modeling matrix.")
    else:
        st.error("Core financial tracking dataset structure error.")

# --- 5. FUTURE EXPANSION HOOKS ---
st.markdown("---")
st.caption("⚓ Developer API Core Integrations Status: Webhook Daemon Listening on `localhost:8000` | Alpaca / Interactive Brokers Sandboxed Core: `Offline`")
