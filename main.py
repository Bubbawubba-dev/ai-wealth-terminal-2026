import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================================================
# MOMENTUM ENGINE v2 (external module)
# =========================================================
from momentum_engine_v2 import (
    analyze_ticker,
    EngineConfig,
)

# =========================================================
# 1. CONFIGURATION & STYLING
# =========================================================

st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

st.markdown(
    """
<style>
.metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: #0f172a; border-radius: 4px 4px 0px 0px; padding: 10px 20px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 2. SECURITY
# =========================================================

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

# =========================================================
# 3. DATA ENGINES
# =========================================================

@st.cache_data(ttl=3600)
def get_base_universe():
    return [
        "ASTS", "ANET", "MRVL", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "ARM", "ZS", "APP", "NASA", "ARMG", "UMAC",
        "RKLB", "SNDK", "CYBR", "INTC", "CIFR", "RDDT", "QUBT", "QBTS", "NOW", "HIVE", "ONDS", "F", "WYFI", "GOOGL",
        "AVGO", "MU", "STX", "QCOM", "TE", "BE", "APLD", "CLSK", "CRWV", "KEEL", "CORZ", "ONDS", "IREN", "NBIS",
        "ENPH", "QCOM", "SMCI", "RGTI", "ASTC", "SHOP", "FJET", "NVDA", "SHAZ", "WOLF", "AVAV", "RCAT", "KTOS", "BA",
    ]


@st.cache_data(ttl=1800)
def fetch_historical_data(tickers, days=730):
    if not tickers:
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        data = yf.download(tickers, start=start_date, group_by="ticker", progress=False)
        if data.empty:
            return pd.DataFrame()
        return data
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_intraday_snapshot(tickers, interval="5m", days=3):
    if not tickers:
        return {}
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        data = yf.download(
            tickers,
            start=start_date,
            interval=interval,
            group_by="ticker",
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            out = {}
            for t in data.columns.get_level_values(0).unique():
                out[t] = data[t].dropna()
            return out
        else:
            return {tickers[0]: data.dropna()}
    except Exception:
        return {}


def compute_market_shock_index(index_df, vix_df=None, breadth_pct=None):
    if index_df is None or index_df.empty:
        return 50

    open_today = index_df["Open"].iloc[0]
    last_close = index_df["Close"].iloc[-1]
    intraday_ret = (last_close - open_today) / open_today * 100

    shock_price = np.interp(intraday_ret, [-4, -2, 0], [100, 80, 40])

    shock_vol = 50
    if vix_df is not None and not vix_df.empty:
        vix_change = (vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-2]) / vix_df["Close"].iloc[-2] * 100
        shock_vol = np.interp(vix_change, [0, 10, 30], [40, 70, 95])

    shock_breadth = 50
    if breadth_pct is not None:
        shock_breadth = np.interp(breadth_pct, [20, 40, 60], [95, 70, 40])

    composite = 0.4 * shock_price + 0.35 * shock_vol + 0.25 * shock_breadth
    return int(np.clip(composite, 0, 100))


def compute_ticker_shock(intraday_df, daily_tail_df):
    if intraday_df is None or intraday_df.empty or daily_tail_df is None or daily_tail_df.empty:
        return {
            "intraday_return_pct": 0.0,
            "daily_vol_pct": 0.0,
            "shock_z": 0.0,
            "shock_score": 50,
        }

    open_today = intraday_df["Open"].iloc[0]
    last_close = intraday_df["Close"].iloc[-1]
    intraday_ret = (last_close - open_today) / open_today * 100

    daily_close = daily_tail_df["Close"]
    daily_ret = daily_close.pct_change().dropna()
    vol = daily_ret.std() * 100 if len(daily_ret) > 5 else 1.0

    shock_z = intraday_ret / (vol if vol > 0 else 1.0)
    shock_score = np.interp(shock_z, [-3, -2, -1, 0], [100, 80, 65, 45])

    return {
        "intraday_return_pct": round(intraday_ret, 2),
        "daily_vol_pct": round(vol, 2),
        "shock_z": round(shock_z, 2),
        "shock_score": int(np.clip(shock_score, 0, 100)),
    }


def compute_uvxy_auto_signal():
    """Compute UVXY auto-signal based on VIX volatility."""
    try:
        vix = yf.download("^VIX", period="10d", interval="1d")["Close"]
        vix3m = yf.download("^VIX3M", period="10d", interval="1d")["Close"]

        if vix.empty or vix3m.empty:
            return {"status": "No Data"}

        vix_now = vix.iloc[-1]
        vix_prev = vix.iloc[-2] if len(vix) > 1 else vix_now
        vix_change = (vix_now - vix_prev) / vix_prev * 100

        term_structure = vix_now - vix3m.iloc[-1]  # backwardation if > 0

        if vix_now < 15:
            regime = "Calm"
        elif vix_now < 20:
            regime = "Elevated"
        elif vix_now < 28:
            regime = "Stress"
        else:
            regime = "Shock"

        uvxy_score = (
            np.interp(vix_now, [12, 20, 28, 40], [10, 40, 70, 95]) * 0.6 +
            np.interp(vix_change, [-5, 0, 5, 10], [10, 40, 70, 90]) * 0.3 +
            (80 if term_structure > 0 else 20) * 0.1
        )
        uvxy_score = int(np.clip(uvxy_score, 0, 100))

        if uvxy_score >= 80:
            auto_signal = "Volatility Shock ⚠️"
        elif uvxy_score >= 60:
            auto_signal = "Volatility Expansion ↑"
        elif uvxy_score <= 30:
            auto_signal = "Volatility Compression ↓"
        else:
            auto_signal = "Neutral / No Edge"

        return {
            "VIX": round(vix_now, 2),
            "VIX Change (%)": round(vix_change, 2),
            "Term Structure": round(term_structure, 2),
            "Regime": regime,
            "UVXY Score": uvxy_score,
            "Auto Signal": auto_signal,
        }

    except Exception:
        return {"status": "Error"}


def compute_uvxy_vix_indicator():
    """Compute UVXY/VIX volatility indicator."""
    try:
        vix = yf.download("^VIX", period="10d", interval="1d")["Close"]
        vix3m = yf.download("^VIX3M", period="10d", interval="1d")["Close"]

        if vix.empty or vix3m.empty:
            return {"status": "No Data"}

        vix_now = vix.iloc[-1]
        vix_prev = vix.iloc[-2] if len(vix) > 1 else vix_now
        vix_change = (vix_now - vix_prev) / vix_prev * 100

        term_structure = vix_now - vix3m.iloc[-1]

        if vix_now < 15:
            regime = "Calm"
        elif vix_now < 20:
            regime = "Elevated"
        elif vix_now < 28:
            regime = "Stress"
        else:
            regime = "Shock"

        uvxy_score = (
            np.interp(vix_now, [12, 20, 28, 40], [10, 40, 70, 95]) * 0.6 +
            np.interp(vix_change, [-5, 0, 5, 10], [10, 40, 70, 90]) * 0.3 +
            (80 if term_structure > 0 else 20) * 0.1
        )
        uvxy_score = int(np.clip(uvxy_score, 0, 100))

        return {
            "VIX": round(vix_now, 2),
            "VIX Change (%)": round(vix_change, 2),
            "Term Structure": round(term_structure, 2),
            "Regime": regime,
            "UVXY Score": uvxy_score,
        }

    except Exception:
        return {"status": "Error"}


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
                "Profit Margin": margin_pct,
            }
        except Exception:
            fundamental_records[ticker] = {
                "Market Cap": "N/A",
                "P/E Ratio": "N/A",
                "Profit Margin": "N/A",
            }
    return fundamental_records


def load_daily_ohlcv(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    df = df.dropna()
    df = df.rename(columns=str.title)
    return df

# =========================================================
# 4. CORE TECHNICAL ENGINES
# =========================================================

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
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
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
        "vol_ratio": float(vol_ratio.iloc[-1]),
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

            close = ticker_df["Close"]
            volume = ticker_df["Volume"]
            high = ticker_df["High"]
            low = ticker_df["Low"]

            perf_20d = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            recent_vol_avg = volume.iloc[-20:-1].mean()
            vol_velocity = volume.iloc[-1] / recent_vol_avg if recent_vol_avg > 0 else 1.0

            tr = np.maximum(
                (high - low),
                np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
            )
            atr_20 = tr.rolling(20).mean().iloc[-1]

            sig = unified_signal(ticker_df)
            structure = classify_structure(sig)

            rankings.append(
                {
                    "Ticker": ticker,
                    "Price": round(close.iloc[-1], 2),
                    "20D Return (%)": round(perf_20d, 2),
                    "Vol Velocity (x)": round(vol_velocity, 2),
                    "ATR (20)": round(atr_20, 2),
                    "TR/ATR Ratio": round(tr.iloc[-1] / atr_20 if atr_20 > 0 else 1.0, 2),
                    "Explosive Flag": structure,
                }
            )
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
        rsi_score = np.nan_to_num(rsi, nan
