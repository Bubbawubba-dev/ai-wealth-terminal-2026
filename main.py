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
        "AVGO", "MU", "STX", "LITE", "BE", "CRWV", "IREN", "CORZ", "TE", "APLD", "CLSK", "KEEL", "WYR"
    ]

@st.cache_data(ttl=3600)
def get_growth_universe():
    # Example growth / AI / large-cap momentum names
    return [
        "NVDA", "TSLA", "META", "AMD", "SMCI",
        "SHOP", "CRWD", "NET", "SNOW", "MSTR"
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

def backtest_swing_strategy(df_history, ticker):
    try:
        available_tickers = df_history.columns.get_level_values(0).unique()
        if ticker not in available_tickers:
            return None

        ticker_df = df_history[ticker].dropna()
        close = ticker_df["Close"]
        high = ticker_df["High"]
        low = ticker_df["Low"]

        if len(close) < 60:  # need enough bars
            return None

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))

        # SMA20
        sma20 = close.rolling(20).mean()

        # Vol ratio ATR5 / ATR20
        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )
        atr5 = tr.rolling(5).mean()
        atr20 = tr.rolling(20).mean()
        vol_ratio_series = atr5 / atr20

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
                trade_ret = (close.iloc[i] - entry_price) / entry_price
                returns.append(trade_ret)
                position = None
                entry_price = None

        if not returns:
            return {
                "Ticker": ticker,
                "Avg Trade Return (%)": 0.0,
                "Win Rate (%)": 0.0,
                "Total Trades": 0
            }

        returns_arr = np.array(returns)
        avg_return = float(np.mean(returns_arr) * 100)
        win_rate = float((np.sum(returns_arr > 0) / len(returns_arr)) * 100)

        return {
            "Ticker": ticker,
            "Avg Trade Return (%)": round(avg_return, 2),
            "Win Rate (%)": round(win_rate, 1),
            "Total Trades": int(len(returns_arr))
        }

    except Exception:
        return None

# --- 4. USER INTERFACE PLATFORM ---
st.title("📈 Wealth Terminal v12.0")

core_universe = get_base_universe()
growth_universe = get_growth_universe()
universe = core_universe  # main working universe for tabs 1 & 3 and sentiment
