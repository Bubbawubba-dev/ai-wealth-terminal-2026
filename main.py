import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Wealth Terminal v12.0", layout="wide", page_icon="📈")

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
    if not tickers:
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
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

# --- SHORT-TERM HELPERS ---

def compute_short_term_momentum(df):
    close = df["Close"]
    volume = df["Volume"]

    ret_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    ret_3d = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) >= 4 else 0
    ret_5d = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) >= 6 else 0

    vol_now = volume.iloc[-1]
    vol_avg = volume.tail(20).mean()
    vol_accel = vol_now / vol_avg if vol_avg > 0 else 1

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rs = gain / loss
    rsi5 = 100 - (100 / (1 + rs.iloc[-1]))

    return {
        "1D": round(ret_1d, 2),
        "3D": round(ret_3d, 2),
        "5D": round(ret_5d, 2),
        "VolAccel": round(vol_accel, 2),
        "RSI5": round(rsi5, 2)
    }

def compute_short_term_levels(df):
    close = df["Close"].iloc[-1]
    recent = df.tail(3)
    swing_high = recent["High"].max()
    swing_low = recent["Low"].min()

    breakout = round(swing_high, 2)
    pb_382 = round(swing_low + 0.382 * (swing_high - swing_low), 2)
    pb_618 = round(swing_low + 0.618 * (swing_high - swing_low), 2)

    return {
        "Breakout": breakout,
        "Pullback_382": pb_382,
        "Pullback_618": pb_618,
        "LastClose": round(close, 2),
    }

def breakout_radar(df_history, universe):
    rows = []
    for ticker in universe:
        try:
            df = df_history[ticker].dropna()
            if len(df) < 5:
                continue

            close = df["Close"].iloc[-1]
            prev_high = df["High"].iloc[-2]

            momentum = compute_short_term_momentum(df)

            if close > prev_high and momentum["VolAccel"] > 1.2:
                rows.append({
                    "Ticker": ticker,
                    "Price": round(close, 2),
                    "Prev High": round(prev_high, 2),
                    "1D Return (%)": momentum["1D"],
                    "3D Return (%)": momentum["3D"],
                    "Volume Accel (x)": momentum["VolAccel"]
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Volume Accel (x)", ascending=False)

def pullback_scanner(df_history, universe):
    rows = []
    for ticker in universe:
        try:
            df = df_history[ticker].dropna()
            if len(df) < 10:
                continue

            recent = df.tail(5)
            swing_high = recent["High"].max()
            swing_low = recent["Low"].min()

            pb_382 = swing_low + 0.382 * (swing_high - swing_low)
            pb_618 = swing_low + 0.618 * (swing_high - swing_low)

            close = df["Close"].iloc[-1]

            if pb_382 <= close <= pb_618:
                rows.append({
                    "Ticker": ticker,
                    "Price": round(close, 2),
                    "Pullback 38.2%": round(pb_382, 2),
                    "Pullback 61.8%": round(pb_618, 2),
                    "Distance to 38.2%": round(close - pb_382, 2),
                    "Distance to 61.8%": round(pb_618 - close, 2)
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Distance to 38.2%", ascending=True)

def build_regime_aware_narrative(
    market_shock,
    ticker_shock,
    trend_phase,
    sentiment_label,
    signal_quality
):
    lines = []

    if market_shock >= 80:
        lines.append("Global regime is in a high-stress shock phase with elevated volatility and broad risk-off flows.")
    elif market_shock >= 60:
        lines.append("Market conditions reflect a stress regime with expanding volatility and defensive positioning.")
    elif market_shock >= 40:
        lines.append("Volatility is elevated, with mixed risk appetite across major indices.")
    else:
        lines.append("Market regime is calm with stable volatility and balanced risk sentiment.")

    if ticker_shock >= 80:
        lines.append("This asset is experiencing outsized intraday stress relative to its normal volatility profile.")
    elif ticker_shock >= 60:
        lines.append("This asset is under moderate intraday pressure, diverging from its typical volatility range.")
    elif ticker_shock >= 40:
        lines.append("Intraday behaviour is within normal bounds, with no abnormal stress signals.")
    else:
        lines.append("Intraday flows are stable and aligned with calm market conditions.")

    if trend_phase == "Short-Term Breakout 🚀":
        lines.append("Structural regime aligns with a short-term breakout phase, favouring momentum continuation setups.")
    elif trend_phase == "Healthy Uptrend 📈":
        lines.append("Structural regime confirms a healthy multi-timeframe uptrend supportive of trend-following strategies.")
    elif trend_phase == "Accumulation ⏳":
        lines.append("Structural regime suggests accumulation behaviour, often preceding more decisive trend moves.")
    else:
        lines.append("Structural regime is neutral with no strong directional bias confirmed.")

    lines.append(f"Sentiment currently reflects **{sentiment_label}**, consistent with the observed technical structure.")

    if signal_quality >= 75:
        lines.append("Signal quality is strong, indicating high alignment across trend, momentum, volatility, and structure.")
    elif signal_quality >= 55:
        lines.append("Signal quality is moderate, with partial alignment across key components.")
    else:
        lines.append("Signal quality is weak, suggesting caution until conditions improve.")

    return lines

# --- REWRITTEN AI ENGINE FOR SHORT-TERM TRADING ---

def build_ai_stock_selection_table(df_history, universe, fundamental_cache):
    rows = []
    if df_history.empty:
        return pd.DataFrame()

    available = df_history.columns.get_level_values(0).unique()
    intraday_snap = fetch_intraday_snapshot(list(available))

    for ticker in universe:
        if ticker not in available:
            continue

        try:
            df = df_history[ticker].dropna()
            if len(df) < 80:
                continue

            intraday_df = intraday_snap.get(ticker, pd.DataFrame())
            daily_tail = df.tail(30)
            shock = compute_ticker_shock(intraday_df, daily_tail)

            st_mom = compute_short_term_momentum(df)
            st_levels = compute_short_term_levels(df)

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

            tactical_block = (
                np.interp(st_mom["1D"], [-6, 0, 3], [20, 60, 90]) * 0.25 +
                np.interp(st_mom["3D"], [-10, 0, 6], [20, 60, 90]) * 0.20 +
                np.interp(st_mom["VolAccel"], [0.5, 1, 2], [30, 60, 95]) * 0.30 +
                np.interp(st_mom["RSI5"], [20, 50, 80], [20, 60, 90]) * 0.25
            )

            shock_adj = np.interp(shock["shock_score"], [40, 70, 100], [1.0, 0.9, 0.75])
            tactical_block *= shock_adj

            swing_block = (
                sent_score * 0.6 +
                np.clip(factor["Composite"], 0, 100) * 0.4
            )

            structural_block = (
                np.interp(factor["6M"], [-20, 0, 30], [25, 55, 90]) * 0.4 +
                (10 * factor["Trend"] + 20 * factor["Stability"]) * 0.3 +
                factor["Quality"] * 0.3
            )

            ai_score = (
                0.55 * tactical_block +
                0.30 * swing_block +
                0.15 * structural_block
            )
            ai_score = float(np.clip(ai_score, 0, 100))

            price = float(df["Close"].iloc[-1])
            atr20 = float(
                np.maximum(
                    (df["High"] - df["Low"]),
                    np.maximum(abs(df["High"] - df["Close"].shift(1)),
                               abs(df["Low"] - df["Close"].shift(1)))
                ).rolling(20).mean().iloc[-1]
            )

            entry = round(price - 0.5 * atr20, 2)
            stop = round(price - 1.2 * atr20, 2)
            target = round(price + 1.5 * atr20, 2)

            rows.append({
                "Ticker": ticker,
                "Price": round(price, 2),

                "AI Score": round(ai_score, 1),
                "Shock Score": shock["shock_score"],
                "Intraday Return (%)": shock["intraday_return_pct"],

                "1D Return (%)": st_mom["1D"],
                "3D Return (%)": st_mom["3D"],
                "5D Return (%)": st_mom["5D"],
                "Volume Accel (x)": st_mom["VolAccel"],
                "RSI(5)": st_mom["RSI5"],

                "Structure": structure,
                "Sentiment Score": sent_score,
                "3M Return (%)": round(factor["3M"], 2),
                "Stability": round(factor["Stability"], 2),
                "Quality": round(factor["Quality"], 2),
                "Value": round(factor["Value"], 4),

                "Market Cap": fundamentals["Market Cap"],
                "P/E Ratio": fundamentals["P/E Ratio"],
                "Profit Margin": fundamentals["Profit Margin"],

                "Breakout Level": st_levels["Breakout"],
                "Pullback 38.2%": st_levels["Pullback_382"],
                "Pullback 61.8%": st_levels["Pullback_618"],
                "Entry Level": entry,
                "Stop Level": stop,
                "Target Level": target,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by="AI Score", ascending=False).reset_index(drop=True)

# --- 4. USER INTERFACE PLATFORM ---

st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()

with st.spinner("Syncing intraday market stress regime..."):
    intraday_index = fetch_intraday_snapshot(["QQQ"]).get("QQQ", pd.DataFrame())
    market_shock = compute_market_shock_index(intraday_index)

if market_shock >= 80:
    color = "🔴"
    label = "Shock / Crash Regime"
elif market_shock >= 60:
    color = "🟠"
    label = "Stress Regime"
elif market_shock >= 40:
    color = "🟡"
    label = "Elevated Volatility"
else:
    color = "🟢"
    label = "Calm / Normal"

st.markdown(
    f"**{color} Market Shock Index: {market_shock} — {label}**  "
    f"&nbsp;&nbsp;_Intraday stress vs recent volatility._"
)

st.sidebar.markdown("### ➕ Add Custom Stocks")

manual_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):",
    placeholder="e.g., TSLA, AAPL, PLTR"
)

manual_list = []
if manual_input:
    manual_list = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

custom_select = st.sidebar.multiselect(
    "Or select from universe:",
    options=universe,
    default=[]
)

user_added_tickers = list(set(manual_list + custom_select))
full_universe = list(set(universe + user_added_tickers))

st.sidebar.success(f"Tracking {len(full_universe)} total tickers")

with st.spinner("Syncing technical historical structures..."):
    historical_data = fetch_historical_data(full_universe)

with st.spinner("Extracting corporate fundamental structures..."):
    fundamental_cache = fetch_fundamental_metrics(full_universe)

tab_momentum, tab_breakout, tab_pullback, tab_sentiment, tab_macro, tab_ai = st.tabs([
    "⚡ Short-Term Momentum",
    "🚀 Breakout Radar",
    "📉 Pullback Scanner",
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

# --- TAB 2: BREAKOUT RADAR ---
with tab_breakout:
    st.subheader("🚀 Breakout Radar — Real-Time High Breakouts + Volume Expansion")
    if not historical_data.empty:
        df_breakout = breakout_radar(historical_data, full_universe)
        if not df_breakout.empty:
            st.dataframe(df_breakout, use_container_width=True, hide_index=True)
        else:
            st.info("No breakout candidates detected at this time.")
    else:
        st.error("Historical data unavailable.")

# --- TAB 3: PULLBACK SCANNER ---
with tab_pullback:
    st.subheader("📉 Pullback Scanner — 38.2% to 61.8% Retracement Zones")
    if not historical_data.empty:
        df_pullback = pullback_scanner(historical_data, full_universe)
        if not df_pullback.empty:
            st.dataframe(df_pullback, use_container_width=True, hide_index=True)
        else:
            st.info("No assets currently in optimal pullback zones.")
    else:
        st.error("Historical data unavailable.")

# You can now re-attach your existing Technical Sentiment, Macro, and AI tab UIs below,
# using the new engines where useful.
