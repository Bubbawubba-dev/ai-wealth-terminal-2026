import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo
from high_movement import build_candidate_display_context, build_high_movement_payload

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
        "ASTS", "ANET", "BZFD", "HUT", "FLEX", "VCYT", "MSFT", "IONQ", "ARM", "ZS", "APP", "NASA", "DPRO", "UMAC",
        "RKLB", "SNDK", "CYBR", "INTC", "CIFR", "RDDT", "QUBT", "QBTS", "SNOW", "HIVE", "ONDS", "F",
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
                out[t] = normalize_intraday_bars(data[t].dropna())
            return out
        else:
            return {tickers[0]: normalize_intraday_bars(data.dropna())}
    except Exception:
        return {}

# =========================================================
# 3A. INTRADAY 5-MIN WRAPPER (NEW)
# =========================================================

@st.cache_data(ttl=120)
def fetch_intraday_5m(tickers, days=2):
    # Always use 5-minute bars for day trading logic
    return fetch_intraday_snapshot(tickers, interval="5m", days=days)


def fetch_single_intraday_5m(ticker, days=2):
    if not ticker or not isinstance(ticker, str):
        return pd.DataFrame()
    snap = fetch_intraday_5m([ticker.upper()], days=days)
    intraday_df = snap.get(ticker.upper(), pd.DataFrame()) if isinstance(snap, dict) else pd.DataFrame()
    if intraday_df is None or intraday_df.empty:
        return pd.DataFrame()
    return normalize_intraday_bars(intraday_df)


MARKET_TZ = "America/New_York"
MARKET_OPEN = dt_time(9, 30)
MARKET_CLOSE = dt_time(16, 0)
INTRADAY_BAR_MINUTES = 5
VALID_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
NO_TRADE_WINDOWS = [
    (dt_time(9, 30), dt_time(9, 45), "Open volatility burst window"),
    (dt_time(12, 0), dt_time(13, 0), "Low-liquidity lunch window"),
    (dt_time(15, 30), dt_time(16, 0), "Late-session drift window"),
]
EDGE_FRICTION_MARGIN_BY_REGIME = {"calm": 8.0, "elevated": 12.0, "stress": 18.0, "shock": 24.0}
MIN_RR_BY_REGIME = {"calm": 1.6, "elevated": 1.8, "stress": 2.0, "shock": 2.2}


def market_regime_from_shock(market_shock):
    if market_shock >= 80:
        return "shock"
    if market_shock >= 60:
        return "stress"
    if market_shock >= 40:
        return "elevated"
    return "calm"


def get_regime_params(market_shock):
    regime = market_regime_from_shock(market_shock)
    table = {
        "calm": {"base_trigger": 0.6, "atr_mult": 0.55, "rv_mult": 0.40, "min_conf": 48},
        "elevated": {"base_trigger": 0.8, "atr_mult": 0.70, "rv_mult": 0.55, "min_conf": 56},
        "stress": {"base_trigger": 1.0, "atr_mult": 0.90, "rv_mult": 0.70, "min_conf": 64},
        "shock": {"base_trigger": 1.2, "atr_mult": 1.10, "rv_mult": 0.90, "min_conf": 72},
    }
    out = table[regime].copy()
    out["regime"] = regime
    return out


def compute_ripster_cloud(close, fast_spans=(8, 21), slow_spans=(34, 55)):
    fast_a = close.ewm(span=fast_spans[0], adjust=False).mean()
    fast_b = close.ewm(span=fast_spans[1], adjust=False).mean()
    slow_a = close.ewm(span=slow_spans[0], adjust=False).mean()
    slow_b = close.ewm(span=slow_spans[1], adjust=False).mean()
    fast_upper = pd.concat([fast_a, fast_b], axis=1).max(axis=1)
    fast_lower = pd.concat([fast_a, fast_b], axis=1).min(axis=1)
    slow_upper = pd.concat([slow_a, slow_b], axis=1).max(axis=1)
    slow_lower = pd.concat([slow_a, slow_b], axis=1).min(axis=1)
    fast_mid = (fast_upper + fast_lower) / 2
    slow_mid = (slow_upper + slow_lower) / 2
    return {
        "fast_upper": fast_upper,
        "fast_lower": fast_lower,
        "slow_upper": slow_upper,
        "slow_lower": slow_lower,
        "fast_mid": fast_mid,
        "slow_mid": slow_mid,
    }


def classify_cloud_state(price, cloud):
    fast_upper = float(cloud["fast_upper"].iloc[-1])
    fast_lower = float(cloud["fast_lower"].iloc[-1])
    slow_upper = float(cloud["slow_upper"].iloc[-1])
    slow_lower = float(cloud["slow_lower"].iloc[-1])
    fast_mid = cloud["fast_mid"]
    slow_mid = cloud["slow_mid"]
    px = float(price.iloc[-1])

    fast_slope = float(fast_mid.diff().tail(5).mean()) if len(fast_mid) >= 6 else 0.0
    slow_slope = float(slow_mid.diff().tail(5).mean()) if len(slow_mid) >= 6 else 0.0

    fast_width = max(fast_upper - fast_lower, 0.0)
    slow_width = max(slow_upper - slow_lower, 0.0)
    close_px = max(px, 1e-9)
    compression = ((fast_width + slow_width) / close_px * 100) < 0.6
    transition = np.sign(fast_slope) != np.sign(slow_slope)

    if px > fast_upper > fast_lower > slow_upper > slow_lower and fast_slope > 0 and slow_slope > 0:
        return "bullish trend"
    if px < fast_lower < fast_upper < slow_lower < slow_upper and fast_slope < 0 and slow_slope < 0:
        return "bearish trend"
    if compression:
        return "compression/chop"
    if transition:
        return "transition"
    return "compression/chop"


def cloud_quality_metrics(close, cloud):
    px = max(float(close.iloc[-1]), 1e-9)
    fast_mid = cloud["fast_mid"]
    slow_mid = cloud["slow_mid"]
    fast_width = cloud["fast_upper"] - cloud["fast_lower"]
    slow_width = cloud["slow_upper"] - cloud["slow_lower"]
    cloud_sep_pct = abs(float(fast_mid.iloc[-1] - slow_mid.iloc[-1])) / px * 100
    slope_strength = (abs(float(fast_mid.diff().tail(5).mean())) + abs(float(slow_mid.diff().tail(5).mean()))) / px * 100
    expansion_now = max(float(fast_width.iloc[-1] + slow_width.iloc[-1]), 0.0)
    expansion_prev = float((fast_width + slow_width).tail(20).mean()) if len(fast_width) >= 5 else expansion_now
    expansion_ratio = expansion_now / (expansion_prev + 1e-9)
    flips = int((np.sign(fast_mid.diff().tail(20)).diff().abs() > 0).sum()) if len(fast_mid) >= 25 else 0
    return {
        "separation_pct": cloud_sep_pct,
        "slope_strength_pct": slope_strength,
        "expansion_ratio": expansion_ratio,
        "flip_count": flips,
    }


def cloud_confidence_score(cloud_state, cloud_metrics):
    sep_score = float(np.interp(cloud_metrics["separation_pct"], [0.05, 0.2, 0.8], [15, 55, 95]))
    slope_score = float(np.interp(cloud_metrics["slope_strength_pct"], [0.01, 0.05, 0.2], [10, 50, 95]))
    expansion_score = float(np.interp(cloud_metrics["expansion_ratio"], [0.7, 1.0, 1.4], [25, 60, 95]))
    raw = sep_score * 0.4 + slope_score * 0.35 + expansion_score * 0.25
    if cloud_state == "compression/chop":
        raw -= 20
    if cloud_state == "transition":
        raw -= 14
    if cloud_metrics["flip_count"] >= 4:
        raw -= 10
    return float(np.clip(raw, 0, 100))


def setup_quality_grade(confidence, cloud_confidence, rr_ratio, net_edge_bps):
    composite = confidence * 0.45 + cloud_confidence * 0.35 + np.interp(rr_ratio, [1.0, 1.6, 2.3], [35, 70, 95]) * 0.2
    if net_edge_bps >= 70 and composite >= 88:
        return "A+"
    if composite >= 76:
        return "A"
    if composite >= 62:
        return "B"
    return "C"


def in_no_trade_window(ts):
    t = ts.time()
    for start, end, label in NO_TRADE_WINDOWS:
        if start <= t < end:
            return True, label
    return False, ""


def mtf_cheatcode_alignment(intraday_close, daily_close):
    intraday_cloud = compute_ripster_cloud(intraday_close)
    daily_cloud = compute_ripster_cloud(daily_close)
    intraday_state = classify_cloud_state(intraday_close, intraday_cloud)
    daily_state = classify_cloud_state(daily_close, daily_cloud)
    intraday_bias = intraday_state in {"bullish trend", "transition"}
    daily_bias = daily_state in {"bullish trend", "transition"}
    weekly_bias = bool(daily_close.tail(5).mean() > daily_close.tail(20).mean()) if len(daily_close) >= 20 else True
    aligned = intraday_bias and daily_bias and weekly_bias
    score = 100 if aligned else (65 if intraday_bias and daily_bias else 35)
    return {
        "aligned": aligned,
        "score": score,
        "intraday_state": intraday_state,
        "daily_state": daily_state,
    }


def derive_higher_timeframe_close(close, step=5):
    if isinstance(close.index, pd.DatetimeIndex):
        higher_tf = close.resample("W-FRI").last().dropna()
        if len(higher_tf) >= 4:
            return higher_tf
    higher_tf = close.iloc[::step].dropna()
    if len(higher_tf) == 0 or higher_tf.index[-1] != close.index[-1]:
        higher_tf = pd.concat([higher_tf, close.tail(1)])
        higher_tf = higher_tf[~higher_tf.index.duplicated(keep="last")]
    return higher_tf.dropna()


def calculate_ripster_sentiment_components(ticker_df, rsi_score, ma_score, vol_score):
    close = ticker_df["Close"].dropna()
    cloud = compute_ripster_cloud(close)
    cloud_state = classify_cloud_state(close, cloud)
    cloud_metrics = cloud_quality_metrics(close, cloud)
    cloud_confidence = cloud_confidence_score(cloud_state, cloud_metrics)

    higher_tf_close = derive_higher_timeframe_close(close)
    higher_tf_cloud = compute_ripster_cloud(higher_tf_close)
    higher_tf_state = classify_cloud_state(higher_tf_close, higher_tf_cloud)
    higher_tf_metrics = cloud_quality_metrics(higher_tf_close, higher_tf_cloud)
    higher_tf_confidence = cloud_confidence_score(higher_tf_state, higher_tf_metrics)
    mtf_cheatcode = mtf_cheatcode_alignment(close, higher_tf_close)

    bullish_states = {"bullish trend", "transition"}
    aligned = cloud_state == "bullish trend" and higher_tf_state == "bullish trend"
    supportive_alignment = cloud_state in bullish_states and higher_tf_state in bullish_states
    higher_tf_ma = higher_tf_close.rolling(min(4, len(higher_tf_close))).mean().iloc[-1]
    higher_tf_trend_up = bool(higher_tf_close.iloc[-1] >= higher_tf_ma) if len(higher_tf_close) else False
    if aligned and higher_tf_trend_up and mtf_cheatcode["aligned"]:
        mtf_alignment_score = 100.0
    elif supportive_alignment and higher_tf_trend_up:
        mtf_alignment_score = max(82.0, float(mtf_cheatcode["score"]) * 0.9)
    elif supportive_alignment:
        mtf_alignment_score = max(68.0, float(mtf_cheatcode["score"]) * 0.8)
    elif "compression/chop" in {cloud_state, higher_tf_state}:
        mtf_alignment_score = 38.0
    else:
        mtf_alignment_score = 18.0

    trend_confirmation_score = float(
        np.clip(
            ma_score * 0.55
            + cloud_confidence * 0.25
            + higher_tf_confidence * 0.20
            + (8 if higher_tf_trend_up else -8),
            0,
            100,
        )
    )

    higher_tf_return = (
        ((higher_tf_close.iloc[-1] - higher_tf_close.iloc[-4]) / higher_tf_close.iloc[-4]) * 100
        if len(higher_tf_close) >= 4 and higher_tf_close.iloc[-4] != 0
        else 0.0
    )
    momentum_confirmation_score = float(
        np.clip(
            rsi_score * 0.75
            + np.interp(higher_tf_return, [-8, 0, 8], [10, 50, 90]) * 0.25,
            0,
            100,
        )
    )

    volatility_context_score = float(np.clip(vol_score * 0.7 + cloud_confidence * 0.3, 0, 100))
    structure_label = classify_structure(unified_signal(ticker_df))
    structure_score = {
        "Short-Term Breakout 🚀": 92.0,
        "Healthy Uptrend 📈": 82.0,
        "Accumulation ⏳": 64.0,
        "Neutral / Wait ⚪": 42.0,
    }.get(structure_label, 50.0)

    ripster_score = int(
        np.average(
            [
                mtf_alignment_score,
                trend_confirmation_score,
                momentum_confirmation_score,
                volatility_context_score,
                structure_score,
                cloud_confidence,
                higher_tf_confidence,
                float(mtf_cheatcode["score"]),
            ],
            weights=[0.22, 0.16, 0.14, 0.10, 0.10, 0.12, 0.08, 0.08],
        )
    )

    return {
        "ripster_mtf_score": ripster_score,
        "ripster_daily_state": cloud_state,
        "ripster_daily_confidence": round(cloud_confidence, 1),
        "ripster_weekly_state": higher_tf_state,
        "ripster_weekly_confidence": round(higher_tf_confidence, 1),
        "ripster_alignment_score": round(mtf_alignment_score, 1),
        "ripster_cheatcode_score": round(float(mtf_cheatcode["score"]), 1),
        "ripster_cheatcode_pass": mtf_cheatcode["aligned"],
        "ripster_trend_score": round(trend_confirmation_score, 1),
        "ripster_momentum_score": round(momentum_confirmation_score, 1),
        "ripster_volatility_score": round(volatility_context_score, 1),
        "ripster_structure_score": round(structure_score, 1),
        "ripster_structure_label": structure_label,
        "ripster_weekly_return_pct": round(higher_tf_return, 2),
    }


def evaluate_breakout_confirmation(close, high, volume):
    if len(close) < 4:
        return {"one_candle_confirmation": False, "failed_breakout": False}
    rolling_high = high.rolling(20).max().shift(1)
    breakout_now = bool(close.iloc[-2] > rolling_high.iloc[-2]) if not np.isnan(rolling_high.iloc[-2]) else False
    one_candle_confirmation = breakout_now and close.iloc[-1] > close.iloc[-2]
    vol_baseline = float(volume.tail(30).mean()) if len(volume) else 0.0
    vol_ok = float(volume.iloc[-1]) >= vol_baseline
    failed_breakout = breakout_now and (close.iloc[-1] < close.iloc[-2] or not vol_ok)
    return {
        "one_candle_confirmation": one_candle_confirmation and vol_ok,
        "failed_breakout": failed_breakout,
    }


def sanitize_universe(tickers):
    cleaned = []
    invalid = []
    duplicates = []
    seen = set()

    for raw in tickers or []:
        t = str(raw).strip().upper()
        if not t:
            continue
        if t in seen:
            duplicates.append(t)
            continue
        seen.add(t)
        if not VALID_TICKER_RE.match(t):
            invalid.append(t)
            continue
        cleaned.append(t)

    return cleaned, sorted(set(invalid)), sorted(set(duplicates))


def normalize_intraday_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(MARKET_TZ)
    out.index = idx
    out = out.sort_index()
    out = out[out.index.dayofweek < 5]
    out = out.between_time(MARKET_OPEN.strftime("%H:%M"), MARKET_CLOSE.strftime("%H:%M"))
    return out.dropna(subset=["Close"])


def compute_previous_session_close(intraday_df, daily_df):
    intraday_df = normalize_intraday_bars(intraday_df)
    if intraday_df.empty:
        if daily_df is None or daily_df.empty:
            return None
        return float(daily_df["Close"].iloc[-1])

    current_session = intraday_df.index[-1].date()
    prev_session_mask = intraday_df.index.date < current_session
    prev_session = intraday_df.loc[prev_session_mask]
    if not prev_session.empty:
        return float(prev_session["Close"].iloc[-1])

    if daily_df is None or daily_df.empty:
        return float(intraday_df["Close"].iloc[-1])

    idx = pd.DatetimeIndex(daily_df.index)
    daily_dates = idx.tz_localize(None) if idx.tz is not None else idx
    before_mask = daily_dates.date < current_session
    if before_mask.any():
        return float(daily_df.loc[before_mask, "Close"].iloc[-1])
    return float(daily_df["Close"].iloc[-1])


def assess_intraday_data_quality(intraday_df, now_ts=None):
    intraday_df = normalize_intraday_bars(intraday_df)
    if intraday_df.empty:
        return {
            "usable": False,
            "fresh": False,
            "stale_minutes": None,
            "missing_ratio": 1.0,
            "warnings": ["No intraday bars available after market-hours filtering."],
            "last_bar": None,
            "bars_in_session": 0,
        }

    now_ny = now_ts or datetime.now(ZoneInfo(MARKET_TZ))
    last_bar = intraday_df.index[-1]
    stale_minutes = (now_ny - last_bar).total_seconds() / 60.0

    session_mask = intraday_df.index.date == last_bar.date()
    session_df = intraday_df.loc[session_mask]

    session_open = datetime.combine(last_bar.date(), MARKET_OPEN, tzinfo=ZoneInfo(MARKET_TZ))
    session_close = datetime.combine(last_bar.date(), MARKET_CLOSE, tzinfo=ZoneInfo(MARKET_TZ))
    expected_end = min(last_bar, session_close)
    expected_idx = pd.date_range(
        start=session_open,
        end=expected_end,
        freq=f"{INTRADAY_BAR_MINUTES}min",
        tz=ZoneInfo(MARKET_TZ),
    )
    expected_bars = max(len(expected_idx), 1)
    missing_ratio = float(max(expected_bars - len(session_df), 0) / expected_bars)

    warnings = []
    fresh = stale_minutes <= 20
    if not fresh:
        warnings.append(f"Stale intraday feed ({stale_minutes:.0f} min old).")
    if missing_ratio > 0.15:
        warnings.append(f"Missing intraday bars detected ({missing_ratio:.0%} of expected bars).")

    return {
        "usable": fresh and missing_ratio <= 0.25,
        "fresh": fresh,
        "stale_minutes": round(stale_minutes, 1),
        "missing_ratio": round(missing_ratio, 3),
        "warnings": warnings,
        "last_bar": last_bar,
        "bars_in_session": int(len(session_df)),
    }


def compute_beta_to_market(ticker_df, market_df, lookback=63):
    if ticker_df is None or ticker_df.empty or market_df is None or market_df.empty:
        return np.nan
    t_ret = ticker_df["Close"].pct_change().tail(lookback).dropna()
    m_ret = market_df["Close"].pct_change().tail(lookback).dropna()
    if t_ret.empty or m_ret.empty:
        return np.nan
    merged = pd.concat([t_ret, m_ret], axis=1, join="inner").dropna()
    if len(merged) < 20:
        return np.nan
    cov = np.cov(merged.iloc[:, 0], merged.iloc[:, 1])[0, 1]
    var = np.var(merged.iloc[:, 1])
    return float(cov / var) if var > 0 else np.nan


def walk_forward_metrics(returns):
    if returns is None or len(returns) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
        }

    r = pd.Series(returns).dropna()
    if r.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
        }

    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    downside = r[r < 0].std() if (r < 0).any() else 0.0
    sharpe = (r.mean() / r.std() * np.sqrt(252)) if r.std() and r.std() > 0 else 0.0
    sortino = (r.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean() * 100),
        "expectancy_pct": float(r.mean() * 100),
        "max_drawdown_pct": float(abs(dd) * 100),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(pf if np.isfinite(pf) else 999.0),
    }


@st.cache_data(ttl=900)
def run_walk_forward_validation(df):
    if df is None or df.empty or len(df) < 180:
        return {"status": "insufficient"}
    required_cols = {"Close", "High", "Low", "Volume"}
    if not required_cols.issubset(df.columns):
        return {"status": "insufficient"}

    close = df["Close"].copy()
    ret_fwd = close.pct_change().shift(-1)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    tr = np.maximum(
        (df["High"] - df["Low"]),
        np.maximum(abs(df["High"] - close.shift(1)), abs(df["Low"] - close.shift(1))),
    )
    atr5 = tr.rolling(5).mean()
    atr20 = tr.rolling(20).mean()
    atr_pct = (atr5 / close * 100).fillna(0)
    vol_ratio = (atr5 / (atr20 + 1e-9)).fillna(1.0)
    cloud = compute_ripster_cloud(close)
    cloud_state = pd.Series(index=close.index, dtype=object)
    for i in range(len(close)):
        sub_close = close.iloc[: i + 1]
        sub_cloud = {k: v.iloc[: i + 1] for k, v in cloud.items()}
        cloud_state.iloc[i] = classify_cloud_state(sub_close, sub_cloud)
    cloud_aligned = cloud_state.eq("bullish trend")
    volume_confirmation = (df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)) > 1.05
    time_window_proxy = atr_pct <= atr_pct.rolling(30).quantile(0.85).fillna(atr_pct.max())

    feat = pd.DataFrame(
        {
            "trend": (close / (sma20 + 1e-9) - 1).fillna(0),
            "mom": ((rsi - 50) / 50).fillna(0),
            "vol": (1.2 - vol_ratio).fillna(0),
            "ret_fwd": ret_fwd,
            "atr_pct": atr_pct,
            "cloud_ok": cloud_aligned.fillna(False),
            "volume_ok": volume_confirmation.fillna(False),
            "time_ok": time_window_proxy.fillna(False),
        }
    ).dropna()
    if len(feat) < 140:
        return {"status": "insufficient"}

    train_size = 126
    test_size = 21
    grid = [
        (0.5, 0.3, 0.2),
        (0.4, 0.4, 0.2),
        (0.3, 0.5, 0.2),
        (0.45, 0.35, 0.20),
        (0.35, 0.45, 0.20),
    ]

    oos_returns = []
    oos_regimes = []
    oos_ablation = {"no_cloud_filter": [], "no_volume_confirmation": [], "no_time_window_filter": []}
    aligned_returns = []
    non_aligned_returns = []

    for start in range(0, len(feat) - train_size - test_size + 1, test_size):
        train = feat.iloc[start:start + train_size]
        test = feat.iloc[start + train_size:start + train_size + test_size]

        best_w = grid[0]
        best_exp = -1e9
        for w in grid:
            s = train["trend"] * w[0] + train["mom"] * w[1] + train["vol"] * w[2]
            trn_rets = train.loc[s > 0.15, "ret_fwd"]
            exp = trn_rets.mean() if len(trn_rets) else -1e9
            if exp > best_exp:
                best_exp = exp
                best_w = w

        score = test["trend"] * best_w[0] + test["mom"] * best_w[1] + test["vol"] * best_w[2]
        take = (score > 0.15) & test["cloud_ok"] & test["volume_ok"] & test["time_ok"]
        oos_returns.extend(list(test.loc[take, "ret_fwd"].dropna()))
        aligned_returns.extend(list(test.loc[take & test["cloud_ok"], "ret_fwd"].dropna()))
        non_aligned_returns.extend(list(test.loc[(score > 0.15) & (~test["cloud_ok"]), "ret_fwd"].dropna()))

        q1, q2, q3 = train["atr_pct"].quantile([0.25, 0.5, 0.75]).tolist()
        for i in test.index[take]:
            a = test.loc[i, "atr_pct"]
            if a <= q1:
                oos_regimes.append(("calm", test.loc[i, "ret_fwd"]))
            elif a <= q2:
                oos_regimes.append(("elevated", test.loc[i, "ret_fwd"]))
            elif a <= q3:
                oos_regimes.append(("stress", test.loc[i, "ret_fwd"]))
            else:
                oos_regimes.append(("shock", test.loc[i, "ret_fwd"]))

        no_cloud = (score > 0.15) & test["volume_ok"] & test["time_ok"]
        no_volume = (score > 0.15) & test["cloud_ok"] & test["time_ok"]
        no_time = (score > 0.15) & test["cloud_ok"] & test["volume_ok"]
        oos_ablation["no_cloud_filter"].extend(list(test.loc[no_cloud, "ret_fwd"].dropna()))
        oos_ablation["no_volume_confirmation"].extend(list(test.loc[no_volume, "ret_fwd"].dropna()))
        oos_ablation["no_time_window_filter"].extend(list(test.loc[no_time, "ret_fwd"].dropna()))

    overall = walk_forward_metrics(oos_returns)
    regime_rows = []
    if oos_regimes:
        reg_df = pd.DataFrame(oos_regimes, columns=["Regime", "ret"])
        for rg, grp in reg_df.groupby("Regime"):
            m = walk_forward_metrics(grp["ret"].tolist())
            regime_rows.append(
                {
                    "Regime": rg,
                    "Trades": m["trades"],
                    "Win Rate (%)": round(m["win_rate"], 1),
                    "Expectancy (%)": round(m["expectancy_pct"], 3),
                }
            )

    base_exp = overall["expectancy_pct"]
    ablation_rows = []
    for k, vals in oos_ablation.items():
        m = walk_forward_metrics(vals)
        ablation_rows.append(
            {
                "Ablation": k.replace("_", " ").title(),
                "Expectancy (%)": round(m["expectancy_pct"], 3),
                "Delta vs Base (pp)": round(m["expectancy_pct"] - base_exp, 3),
            }
        )
    cloud_alignment_df = pd.DataFrame(
        [
            {"Subset": "Cloud Aligned", **walk_forward_metrics(aligned_returns)},
            {"Subset": "Cloud Non-Aligned", **walk_forward_metrics(non_aligned_returns)},
        ]
    )
    if not cloud_alignment_df.empty:
        cloud_alignment_df["Expectancy Lift (pp)"] = cloud_alignment_df["expectancy_pct"] - float(
            cloud_alignment_df.loc[cloud_alignment_df["Subset"] == "Cloud Non-Aligned", "expectancy_pct"].iloc[0]
            if (cloud_alignment_df["Subset"] == "Cloud Non-Aligned").any()
            else 0.0
        )
        cloud_alignment_df = cloud_alignment_df.rename(
            columns={
                "trades": "Trades",
                "win_rate": "Win Rate (%)",
                "expectancy_pct": "Expectancy (%)",
                "profit_factor": "Profit Factor",
            }
        )

    return {
        "status": "ok",
        "overall": overall,
        "regime_df": pd.DataFrame(regime_rows),
        "ablation_df": pd.DataFrame(ablation_rows),
        "cloud_alignment_df": cloud_alignment_df,
    }


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
                "Profit Margin": margin_pct,
            }
        except Exception:
            fundamental_records[ticker] = {
                "Market Cap": "N/A",
                "P/E Ratio": "N/A",
                "Profit Margin": "N/A",
            }
    return fundamental_records

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
        rsi_score = np.nan_to_num(rsi, nan=50.0)

        sma_20 = close.rolling(20).mean().iloc[-1]
        current_price = close.iloc[-1]
        price_to_sma_pct = ((current_price - sma_20) / sma_20) * 100
        ma_score = np.interp(price_to_sma_pct, [-10, 10], [0, 100])

        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
        )
        atr_5 = tr.rolling(5).mean().iloc[-1]
        atr_20 = tr.rolling(20).mean().iloc[-1]
        vol_ratio = atr_5 / atr_20 if atr_20 > 0 else 1
        vol_score = np.interp(vol_ratio, [0.8, 1.5], [80, 20])
        ripster_metrics = calculate_ripster_sentiment_components(ticker_df, rsi_score, ma_score, vol_score)

        composite_score = int(
            np.average(
                [rsi_score, ma_score, vol_score, ripster_metrics["ripster_mtf_score"]],
                weights=[0.3, 0.25, 0.15, 0.3],
            )
        )

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
                "volatility_ratio": round(vol_ratio, 2),
                **ripster_metrics,
            },
        }

    except Exception as e:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "score": 50,
            "label": "Neutral (Insufficient Data)",
            "error": str(e),
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
            "error": sentiment_result.get("error"),
        }
    except Exception as e:
        return {"status": "Error", "score": 50, "label": "Error", "error": str(e)}


@st.cache_data(ttl=900)
def calculate_macro_trends(df_history, tickers, fundamental_data):
    macro_data = []
    if df_history.empty:
        return pd.DataFrame()

    available_tickers = df_history.columns.get_level_values(0).unique()

    for ticker in tickers:
        try:
            if ticker not in available_tickers:
                continue

            df = df_history[ticker].dropna()
            close = df["Close"]
            if len(close) == 0:
                continue

            if len(close) < 200:
                sma_50 = close.rolling(50).mean().iloc[-1]
                sma_200 = sma_50
            else:
                sma_50 = close.rolling(50).mean().iloc[-1]
                sma_200 = close.rolling(200).mean().iloc[-1]

            current_price = close.iloc[-1]
            dist_from_sma200 = ((current_price - sma_200) / sma_200) * 100 if sma_200 != 0 else 0.0

            perf_6month = (
                (current_price - close.iloc[-126]) / close.iloc[-126] * 100
                if len(close) >= 126
                else 0.0
            )

            sig = unified_signal(df)
            regime = classify_structure(sig)

            f = fundamental_data.get(
                ticker,
                {
                    "Market Cap": "N/A",
                    "P/E Ratio": "N/A",
                    "Profit Margin": "N/A",
                },
            )

            macro_data.append(
                {
                    "Ticker": ticker,
                    "Current Price": round(current_price, 2),
                    "Market Cap": f["Market Cap"],
                    "P/E Ratio": f["P/E Ratio"],
                    "Profit Margin": f["Profit Margin"],
                    "Dist. from 200D (%)": round(dist_from_sma200, 2),
                    "6M Return (%)": round(perf_6month, 2),
                    "Macro Structure": regime,
                }
            )

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
            3
            if close.iloc[-1] > sma50 > sma200
            else 1
            if close.iloc[-1] > sma200
            else -1
            if sma50 > sma200
            else -3
        )

        high = df["High"]
        low = df["Low"]
        tr = np.maximum(
            (high - low),
            np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
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
            (ret_3m * 0.25)
            + (trend_strength * 10 * 0.25)
            + (stability * 20 * 0.25)
            + (quality * 0.15)
            + (value * 50 * 0.10)
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
            "Composite": composite,
        }

    except Exception:
        return None

# =========================================================
# 5. SHORT-TERM ENGINES (BREAKOUT / PULLBACK / MOMENTUM)
# =========================================================

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
        "RSI5": round(rsi5, 2),
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
                rows.append(
                    {
                        "Ticker": ticker,
                        "Price": round(close, 2),
                        "Prev High": round(prev_high, 2),
                        "1D Return (%)": momentum["1D"],
                        "3D Return (%)": momentum["3D"],
                        "Volume Accel (x)": momentum["VolAccel"],
                    }
                )
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
                rows.append(
                    {
                        "Ticker": ticker,
                        "Price": round(close, 2),
                        "Pullback 38.2%": round(pb_382, 2),
                        "Pullback 61.8%": round(pb_618, 2),
                        "Distance to 38.2%": round(close - pb_382, 2),
                        "Distance to 61.8%": round(pb_618 - close, 2),
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Distance to 38.2%", ascending=True)

# =========================================================
# 6. SIGNAL QUALITY & REGIME-AWARE NARRATIVE
# =========================================================

def compute_signal_quality_and_narrative(
    close,
    sma20,
    rsi_series,
    vol_ratio_series,
    returns,
    win_rate,
    avg_return,
    trend_phase,
):
    latest_price = float(close.iloc[-1])
    latest_sma20 = float(sma20.iloc[-1])
    latest_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    latest_vol = float(vol_ratio_series.iloc[-1]) if not np.isnan(vol_ratio_series.iloc[-1]) else 1.0

    if latest_sma20 > 0:
        dist_sma20_pct = (latest_price - latest_sma20) / latest_sma20 * 100
    else:
        dist_sma20_pct = 0.0

    trend_score = float(
        np.interp(
            dist_sma20_pct,
            [-5, 0, 5, 10],
            [0, 40, 70, 100],
        )
    )

    momentum_score = float(
        np.interp(
            latest_rsi,
            [30, 45, 55, 70],
            [0, 40, 70, 100],
        )
    )

    vol_score = float(
        np.interp(
            latest_vol,
            [0.6, 0.9, 1.1, 1.6],
            [20, 80, 60, 20],
        )
    )

    if returns:
        win_component = float(
            np.interp(
                win_rate,
                [30, 50, 70],
                [20, 60, 100],
            )
        )
        ret_component = float(
            np.interp(
                avg_return,
                [-2, 0, 2],
                [20, 60, 100],
            )
        )
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
        trend_score * 0.30
        + momentum_score * 0.25
        + vol_score * 0.20
        + backtest_score * 0.15
        + structure_score * 0.10
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


def build_regime_aware_narrative(
    market_shock,
    ticker_shock,
    trend_phase,
    sentiment_label,
    signal_quality,
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

# =========================================================
# 7. REWRITTEN AI ENGINE FOR SHORT-TERM TRADING
# =========================================================

# =========================================================
# 7A. INTRADAY DAY-TRADING ENGINE (NEW)
# =========================================================

def compute_intraday_trade_plan(intraday_df, daily_df, market_shock=50):
    params = get_regime_params(market_shock)
    empty = {
        "prev_close": None,
        "last_price": None,
        "buy_trigger": False,
        "sell_trigger": False,
        "distance_from_prev_close_pct": 0.0,
        "momentum_score": 50.0,
        "exit_warning": True,
        "trigger_threshold_pct": params["base_trigger"],
        "momentum_alignment": False,
        "trend_filter_pass": False,
        "cloud_state": "compression/chop",
        "cloud_confidence": 0.0,
        "mtf_cheatcode_pass": False,
        "one_candle_confirmation": False,
        "failed_breakout": False,
        "regime": params["regime"],
        "confidence_score": 0.0,
        "expected_move_pct": 0.0,
        "stop_loss_pct": 0.0,
        "target_pct": 0.0,
        "reward_to_risk": 0.0,
        "trailing_stop_pct": 0.0,
        "max_hold_bars": 0,
        "total_cost_bps": 0.0,
        "net_edge_bps": -999.0,
        "friction_margin_bps": EDGE_FRICTION_MARGIN_BY_REGIME.get(params["regime"], 10.0),
        "volume_confirmation_pass": False,
        "invalidation_trigger": "N/A",
        "setup_grade": "C",
        "decision_tags": "",
        "do_not_trade": True,
        "do_not_trade_reason": "Missing intraday or daily data",
    }
    if intraday_df is None or intraday_df.empty or daily_df is None or daily_df.empty:
        return empty

    intraday_df = normalize_intraday_bars(intraday_df)
    if intraday_df.empty:
        return empty

    prev_close = compute_previous_session_close(intraday_df, daily_df)
    last_price = float(intraday_df["Close"].iloc[-1])
    if prev_close is None or prev_close <= 0:
        prev_close = last_price

    distance_pct = (last_price - prev_close) / prev_close * 100
    close = intraday_df["Close"]
    high = intraday_df["High"]
    low = intraday_df["Low"]
    volume = intraday_df["Volume"]
    cloud = compute_ripster_cloud(close)
    cloud_state = classify_cloud_state(close, cloud)
    cloud_metrics = cloud_quality_metrics(close, cloud)
    cloud_confidence = cloud_confidence_score(cloud_state, cloud_metrics)
    mtf = mtf_cheatcode_alignment(close, daily_df["Close"])

    ret_5m = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0.0
    ret_15m = ((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100) if len(close) >= 4 else 0.0
    ret_30m = ((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7] * 100) if len(close) >= 7 else 0.0
    momentum_alignment = bool(ret_5m > 0 and ret_15m > 0 and ret_30m > 0)

    vol_now = float(volume.iloc[-1]) if len(volume) else 0.0
    vol_avg = float(volume.tail(30).mean()) if len(volume) else 0.0
    vol_accel = vol_now / vol_avg if vol_avg > 0 else 1.0

    tr = np.maximum(
        (high - low),
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )
    atr5 = float(tr.rolling(5).mean().iloc[-1]) if len(tr) >= 5 else float(tr.mean())
    atr_pct = (atr5 / last_price * 100) if last_price > 0 else 0.0
    realized_vol = float(close.pct_change().tail(12).std() * 100 * np.sqrt(12)) if len(close) >= 12 else 0.0

    vwap = float((close * volume).cumsum().iloc[-1] / (volume.cumsum().iloc[-1] + 1e-9))
    daily_sma20 = float(daily_df["Close"].rolling(20).mean().iloc[-1]) if len(daily_df) >= 20 else float(daily_df["Close"].iloc[-1])
    trend_filter_pass = bool(last_price > vwap and daily_df["Close"].iloc[-1] > daily_sma20)
    volume_confirmation_pass = bool(vol_accel >= (1.05 if params["regime"] in {"calm", "elevated"} else 1.15))
    breakout_flags = evaluate_breakout_confirmation(close, high, volume)
    one_candle_confirmation = breakout_flags["one_candle_confirmation"]
    failed_breakout = breakout_flags["failed_breakout"]
    in_window, window_reason = in_no_trade_window(intraday_df.index[-1])

    trigger_threshold = max(
        params["base_trigger"],
        atr_pct * params["atr_mult"],
        realized_vol * params["rv_mult"],
    )

    vol_spike = atr_pct > (2.5 if params["regime"] in {"stress", "shock"} else 3.5)
    low_volume = vol_accel < (0.75 if params["regime"] in {"calm", "elevated"} else 0.9)

    raw_mom = (ret_5m * 0.25) + (ret_15m * 0.35) + (ret_30m * 0.25) + ((vol_accel - 1.0) * 100 * 0.15)
    if vol_spike:
        raw_mom -= 20
    if low_volume:
        raw_mom -= 12
    momentum_score = float(np.clip(raw_mom + 50, 0, 100))

    confidence = float(
        np.clip(
            (abs(distance_pct) / (trigger_threshold + 1e-9)) * 35
            + (15 if momentum_alignment else 0)
            + (15 if trend_filter_pass else 0)
            + (18 if mtf["aligned"] else 0)
            + np.interp(vol_accel, [0.5, 1.0, 2.0], [5, 12, 18])
            + np.interp(cloud_confidence, [20, 60, 90], [0, 8, 18])
            - (12 if vol_spike else 0)
            - (10 if low_volume else 0),
            0,
            100,
        )
    )

    slippage_bps = 8.0 if params["regime"] in {"stress", "shock"} else 5.0
    spread_bps = 6.0 if vol_spike else 3.0
    commission_bps = 1.0
    total_cost_bps = slippage_bps + spread_bps + commission_bps
    expected_move_pct = max(trigger_threshold, atr_pct * 1.2)
    net_edge_bps = expected_move_pct * 100 - total_cost_bps
    friction_margin_bps = EDGE_FRICTION_MARGIN_BY_REGIME.get(params["regime"], 10.0)
    stop_loss_pct = max(trigger_threshold * 0.8, atr_pct * 0.9, 0.5)
    target_pct = max(trigger_threshold * 1.5, atr_pct * 1.3, 0.8)
    reward_to_risk = target_pct / (stop_loss_pct + 1e-9)
    min_rr = MIN_RR_BY_REGIME.get(params["regime"], 1.6)
    trailing_stop_pct = max(stop_loss_pct * 0.7, 0.4)
    max_hold_bars = 18 if params["regime"] in {"stress", "shock"} else 30
    stop_valid = bool(stop_loss_pct >= max(atr_pct * 0.8, 0.5))
    cloud_alignment = cloud_state == "bullish trend" and cloud_confidence >= 55
    edge_gt_friction = net_edge_bps > friction_margin_bps
    high_conviction_exception = confidence >= 85 and cloud_confidence >= 70 and mtf["aligned"]

    reasons = []
    if confidence < params["min_conf"]:
        reasons.append("Low confidence")
    if low_volume:
        reasons.append("Low intraday liquidity")
    if vol_spike:
        reasons.append("Volatility spike")
    if not cloud_alignment:
        reasons.append("Cloud misalignment")
    if not mtf["aligned"]:
        reasons.append("MTF cheatcode misalignment")
    if not volume_confirmation_pass:
        reasons.append("No volume confirmation")
    if not one_candle_confirmation:
        reasons.append("No one-candle confirmation")
    if failed_breakout:
        reasons.append("Failed breakout invalidation")
    if not stop_valid:
        reasons.append("Stop invalid vs invalidation zone")
    if reward_to_risk < min_rr:
        reasons.append(f"R/R below regime threshold ({min_rr:.1f})")
    if not edge_gt_friction:
        reasons.append("Edge does not clear friction margin")
    if in_window and not high_conviction_exception:
        reasons.append(window_reason)
    if net_edge_bps <= 0:
        reasons.append("Costs exceed expected edge")

    buy_trigger = bool(
        distance_pct >= trigger_threshold
        and momentum_alignment
        and trend_filter_pass
        and cloud_alignment
        and mtf["aligned"]
        and one_candle_confirmation
        and volume_confirmation_pass
        and not failed_breakout
        and confidence >= params["min_conf"]
        and reward_to_risk >= min_rr
        and edge_gt_friction
        and stop_valid
        and (not in_window or high_conviction_exception)
        and not vol_spike
        and not low_volume
    )
    sell_trigger = bool(
        failed_breakout
        or distance_pct <= -0.6 * trigger_threshold
        or (last_price < vwap and ret_5m < 0)
        or confidence < (params["min_conf"] - 10)
    )
    do_not_trade = len(reasons) > 0
    exit_warning = sell_trigger or do_not_trade
    invalidation_trigger = "Failed breakout" if failed_breakout else ("VWAP breakdown" if last_price < vwap else "Trend failure")
    setup_grade = setup_quality_grade(confidence, cloud_confidence, reward_to_risk, net_edge_bps)
    decision_tags = ",".join(
        [
            f"setup:breakout-continuation",
            f"regime:{params['regime']}",
            f"cloud:{cloud_state}",
            f"mtf:{'pass' if mtf['aligned'] else 'fail'}",
            f"entry:{'allowed' if buy_trigger and not do_not_trade else 'blocked'}",
        ]
    )

    return {
        "prev_close": round(float(prev_close), 2),
        "last_price": round(last_price, 2),
        "buy_trigger": buy_trigger,
        "sell_trigger": sell_trigger,
        "distance_from_prev_close_pct": round(distance_pct, 2),
        "momentum_score": round(momentum_score, 1),
        "exit_warning": exit_warning,
        "trigger_threshold_pct": round(trigger_threshold, 2),
        "momentum_alignment": momentum_alignment,
        "trend_filter_pass": trend_filter_pass,
        "cloud_state": cloud_state,
        "cloud_confidence": round(cloud_confidence, 1),
        "cloud_separation_pct": round(cloud_metrics["separation_pct"], 3),
        "cloud_slope_strength_pct": round(cloud_metrics["slope_strength_pct"], 3),
        "cloud_expansion_ratio": round(cloud_metrics["expansion_ratio"], 3),
        "mtf_cheatcode_pass": mtf["aligned"],
        "mtf_cheatcode_score": mtf["score"],
        "mtf_intraday_state": mtf["intraday_state"],
        "mtf_daily_state": mtf["daily_state"],
        "one_candle_confirmation": one_candle_confirmation,
        "failed_breakout": failed_breakout,
        "volume_confirmation_pass": volume_confirmation_pass,
        "regime": params["regime"],
        "confidence_score": round(confidence, 1),
        "expected_move_pct": round(expected_move_pct, 2),
        "atr_context_pct": round(atr_pct, 2),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "target_pct": round(target_pct, 2),
        "reward_to_risk": round(reward_to_risk, 2),
        "trailing_stop_pct": round(trailing_stop_pct, 2),
        "max_hold_bars": int(max_hold_bars),
        "slippage_bps": round(slippage_bps, 1),
        "spread_bps": round(spread_bps, 1),
        "commission_bps": round(commission_bps, 1),
        "total_cost_bps": round(total_cost_bps, 1),
        "net_edge_bps": round(net_edge_bps, 1),
        "friction_margin_bps": round(friction_margin_bps, 1),
        "invalidation_trigger": invalidation_trigger,
        "setup_grade": setup_grade,
        "decision_tags": decision_tags,
        "do_not_trade": do_not_trade,
        "do_not_trade_reason": ", ".join(reasons) if reasons else "",
    }


def build_ai_stock_selection_table(
    df_history,
    universe,
    fundamental_cache,
    market_shock=50,
    min_price=5.0,
    min_dollar_volume=2_000_000,
):
    if df_history.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    rows = []
    diagnostics = []
    audit_rows = []
    regime = market_regime_from_shock(market_shock)
    account_equity = 100000.0
    risk_per_trade = 0.01
    max_positions = 5
    max_daily_loss_pct = 2.0
    daily_risk_budget = account_equity * (max_daily_loss_pct / 100)
    strategy_kill_switch = False
    kill_switch_reason = ""
    projected_daily_risk_used = 0.0
    consecutive_loss_proxy = 0
    universe_clean, invalid_tickers, duplicate_tickers = sanitize_universe(universe)

    available = df_history.columns.get_level_values(0).unique()
    intraday_snap = fetch_intraday_5m(list(set(universe_clean) & set(available)))
    qqq_daily = df_history["QQQ"].dropna() if "QQQ" in available else pd.DataFrame()

    for t in invalid_tickers:
        diagnostics.append({"Ticker": t, "Status": "Rejected", "Reason": "Invalid ticker format"})
    for t in duplicate_tickers:
        diagnostics.append({"Ticker": t, "Status": "Ignored", "Reason": "Duplicate ticker"})

    for ticker in universe_clean:
        if ticker not in available:
            diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": "Ticker missing from historical dataset"})
            continue

        try:
            df = df_history[ticker].dropna()
            if len(df) < 120:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": "Insufficient daily history"})
                continue

            price = float(df["Close"].iloc[-1])
            if price < min_price:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": f"Price below minimum (${min_price:.2f})"})
                continue

            avg_dollar_volume = float((df["Close"] * df["Volume"]).tail(20).mean())
            if avg_dollar_volume < min_dollar_volume:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": "Insufficient liquidity (20D dollar volume)"})
                continue

            intraday_df = intraday_snap.get(ticker, pd.DataFrame())
            dq = assess_intraday_data_quality(intraday_df)
            if not dq["usable"]:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": "; ".join(dq["warnings"]) or "Intraday quality check failed"})
                continue

            intraday_df = normalize_intraday_bars(intraday_df)
            daily_tail = df.tail(60)
            shock = compute_ticker_shock(intraday_df, daily_tail)
            st_mom = compute_short_term_momentum(df)
            sig = unified_signal(df)
            structure = classify_structure(sig)
            sentiment = calculate_advanced_sentiment(df_history, ticker)
            sent_score = sentiment.get("score", 50)
            fundamentals = fundamental_cache.get(
                ticker,
                {"Market Cap": "N/A", "P/E Ratio": "N/A", "Profit Margin": "N/A"},
            )

            intraday_plan = compute_intraday_trade_plan(intraday_df, daily_tail, market_shock=market_shock)
            beta = compute_beta_to_market(df, qqq_daily)
            if np.isfinite(beta) and abs(beta) > 2.2:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": f"Excessive beta exposure ({beta:.2f})"})
                continue

            if intraday_plan["setup_grade"] == "C" or intraday_plan["failed_breakout"]:
                consecutive_loss_proxy += 1
            else:
                consecutive_loss_proxy = 0
            if consecutive_loss_proxy >= 3:
                strategy_kill_switch = True
                kill_switch_reason = "Kill-switch: consecutive failed/low-quality setups"

            projected_risk = account_equity * risk_per_trade
            if projected_daily_risk_used + projected_risk > daily_risk_budget:
                strategy_kill_switch = True
                kill_switch_reason = "Kill-switch: daily risk budget exhausted"

            if strategy_kill_switch:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": kill_switch_reason})
                audit_rows.append(
                    {
                        "Ticker": ticker,
                        "Decision": "Rejected",
                        "Setup Type": "breakout-continuation",
                        "Regime": intraday_plan["regime"],
                        "Reason to Enter": "N/A",
                        "Reason to Avoid": kill_switch_reason,
                        "Invalidation Trigger": intraday_plan["invalidation_trigger"],
                        "Decision Tags": intraday_plan["decision_tags"],
                        "Setup Grade": intraday_plan["setup_grade"],
                    }
                )
                continue

            if intraday_plan["do_not_trade"]:
                diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": intraday_plan["do_not_trade_reason"] or "Do-not-trade rule"})
                audit_rows.append(
                    {
                        "Ticker": ticker,
                        "Decision": "Rejected",
                        "Setup Type": "breakout-continuation",
                        "Regime": intraday_plan["regime"],
                        "Reason to Enter": "N/A",
                        "Reason to Avoid": intraday_plan["do_not_trade_reason"] or "Do-not-trade rule",
                        "Invalidation Trigger": intraday_plan["invalidation_trigger"],
                        "Decision Tags": intraday_plan["decision_tags"],
                        "Setup Grade": intraday_plan["setup_grade"],
                        "Confidence": intraday_plan["confidence_score"],
                        "Reason": intraday_plan["do_not_trade_reason"] or "Do-not-trade rule",
                        "Shock Score": shock["shock_score"],
                    }
                )
                continue
            projected_daily_risk_used += projected_risk

            features = {
                "f_intraday": intraday_plan["momentum_score"],
                "f_sent": sent_score,
                "f_shock": 100 - shock["shock_score"],
                "f_mom1d": np.interp(st_mom["1D"], [-6, 0, 6], [20, 55, 90]),
                "f_mom3d": np.interp(st_mom["3D"], [-10, 0, 10], [20, 55, 90]),
                "f_trend": 85 if structure in {"Short-Term Breakout 🚀", "Healthy Uptrend 📈"} else 45,
                "f_cloud": intraday_plan["cloud_confidence"],
            }

            rows.append(
                {
                    "Ticker": ticker,
                    "Price": round(price, 2),
                    "Shock Score": shock["shock_score"],
                    "Intraday Return (%)": shock["intraday_return_pct"],
                    "Short-Term Structure": structure,
                    "Sentiment Score": sent_score,
                    "Prev Close": intraday_plan["prev_close"],
                    "Last Price (5m)": intraday_plan["last_price"],
                    "Δ vs Prev Close (%)": intraday_plan["distance_from_prev_close_pct"],
                    "Vol-Adj Trigger (%)": intraday_plan["trigger_threshold_pct"],
                    "Buy Trigger": intraday_plan["buy_trigger"],
                    "Sell Trigger": intraday_plan["sell_trigger"],
                    "Momentum Alignment": intraday_plan["momentum_alignment"],
                    "Trend Filter Pass": intraday_plan["trend_filter_pass"],
                    "Cloud State": intraday_plan["cloud_state"],
                    "Cloud Confidence": intraday_plan["cloud_confidence"],
                    "Cloud Separation (%)": intraday_plan["cloud_separation_pct"],
                    "Cloud Slope Strength (%)": intraday_plan["cloud_slope_strength_pct"],
                    "Cloud Expansion Ratio": intraday_plan["cloud_expansion_ratio"],
                    "MTF Cheatcode Pass": intraday_plan["mtf_cheatcode_pass"],
                    "MTF Cheatcode Score": intraday_plan["mtf_cheatcode_score"],
                    "Volume Confirmation": intraday_plan["volume_confirmation_pass"],
                    "One-Candle Confirmation": intraday_plan["one_candle_confirmation"],
                    "Failed Breakout": intraday_plan["failed_breakout"],
                    "Intraday Momentum Score": intraday_plan["momentum_score"],
                    "Confidence Score": intraday_plan["confidence_score"],
                    "Expected Move (%)": intraday_plan["expected_move_pct"],
                    "ATR Context (%)": intraday_plan["atr_context_pct"],
                    "Stop Loss (%)": intraday_plan["stop_loss_pct"],
                    "Target (%)": intraday_plan["target_pct"],
                    "Reward/Risk": intraday_plan["reward_to_risk"],
                    "Trailing Stop (%)": intraday_plan["trailing_stop_pct"],
                    "Max Hold (5m bars)": intraday_plan["max_hold_bars"],
                    "Total Costs (bps)": intraday_plan["total_cost_bps"],
                    "Net Edge (bps)": intraday_plan["net_edge_bps"],
                    "Friction Margin (bps)": intraday_plan["friction_margin_bps"],
                    "Invalidation Trigger": intraday_plan["invalidation_trigger"],
                    "Setup Grade": intraday_plan["setup_grade"],
                    "Decision Tags": intraday_plan["decision_tags"],
                    "Regime": intraday_plan["regime"],
                    "Avg 20D $Vol": round(avg_dollar_volume, 0),
                    "Data Stale (min)": dq["stale_minutes"],
                    "Missing Bars (%)": round(dq["missing_ratio"] * 100, 1),
                    "Market Beta (63D)": round(beta, 2) if np.isfinite(beta) else np.nan,
                    "Market Cap": fundamentals["Market Cap"],
                    "P/E Ratio": fundamentals["P/E Ratio"],
                    "Profit Margin": fundamentals["Profit Margin"],
                    "_f_intraday": features["f_intraday"],
                    "_f_sent": features["f_sent"],
                    "_f_shock": features["f_shock"],
                    "_f_mom1d": features["f_mom1d"],
                    "_f_mom3d": features["f_mom3d"],
                    "_f_trend": features["f_trend"],
                    "_f_cloud": features["f_cloud"],
                }
            )

            audit_rows.append(
                {
                    "Ticker": ticker,
                    "Decision": "Accepted",
                    "Setup Type": "breakout-continuation",
                    "Regime": intraday_plan["regime"],
                    "Reason to Enter": "Cloud aligned + MTF pass + volume and one-candle confirmation",
                    "Reason to Avoid": "",
                    "Invalidation Trigger": intraday_plan["invalidation_trigger"],
                    "Decision Tags": intraday_plan["decision_tags"],
                    "Setup Grade": intraday_plan["setup_grade"],
                    "Confidence": intraday_plan["confidence_score"],
                    "Cloud Confidence": intraday_plan["cloud_confidence"],
                    "Shock Score": shock["shock_score"],
                    "Net Edge (bps)": intraday_plan["net_edge_bps"],
                    "Reason": "Passed all filters",
                }
            )
        except Exception as e:
            diagnostics.append({"Ticker": ticker, "Status": "Rejected", "Reason": f"Computation error: {e}"})
            audit_rows.append({"Ticker": ticker, "Decision": "Rejected", "Reason": f"Error: {e}"})

    if not rows:
        return pd.DataFrame(), pd.DataFrame(diagnostics), pd.DataFrame(audit_rows), {
            "invalid_tickers": invalid_tickers,
            "duplicate_tickers": duplicate_tickers,
            "input_count": len(universe or []),
            "clean_count": len(universe_clean),
        }

    score_df = pd.DataFrame(rows)
    score_cols = ["_f_intraday", "_f_sent", "_f_shock", "_f_mom1d", "_f_mom3d", "_f_trend", "_f_cloud"]
    for c in score_cols:
        mu = score_df[c].mean()
        sd = score_df[c].std()
        score_df[f"z_{c}"] = 0.0 if sd == 0 or np.isnan(sd) else (score_df[c] - mu) / sd

    # Regime-aware calibrated blend on normalized cross-sectional features.
    params = get_regime_params(market_shock)
    if params["regime"] in {"stress", "shock"}:
        w = {"_f_intraday": 0.22, "_f_sent": 0.08, "_f_shock": 0.22, "_f_mom1d": 0.08, "_f_mom3d": 0.08, "_f_trend": 0.14, "_f_cloud": 0.18}
    else:
        w = {"_f_intraday": 0.24, "_f_sent": 0.12, "_f_shock": 0.12, "_f_mom1d": 0.12, "_f_mom3d": 0.09, "_f_trend": 0.14, "_f_cloud": 0.17}

    score_df["AI Score"] = (
        score_df["z__f_intraday"] * w["_f_intraday"]
        + score_df["z__f_sent"] * w["_f_sent"]
        + score_df["z__f_shock"] * w["_f_shock"]
        + score_df["z__f_mom1d"] * w["_f_mom1d"]
        + score_df["z__f_mom3d"] * w["_f_mom3d"]
        + score_df["z__f_trend"] * w["_f_trend"]
        + score_df["z__f_cloud"] * w["_f_cloud"]
    )
    sc_min = float(score_df["AI Score"].min())
    sc_max = float(score_df["AI Score"].max())
    if sc_max > sc_min:
        score_df["AI Score"] = np.interp(score_df["AI Score"], [sc_min, sc_max], [35, 95])
    else:
        score_df["AI Score"] = 65.0

    min_conf = params["min_conf"]
    score_df = score_df[score_df["Confidence Score"] >= min_conf].copy()
    if score_df.empty:
        diagnostics.append({"Ticker": "*", "Status": "Rejected", "Reason": "All symbols failed confidence threshold for current regime."})
        return pd.DataFrame(), pd.DataFrame(diagnostics), pd.DataFrame(audit_rows), {
            "invalid_tickers": invalid_tickers,
            "duplicate_tickers": duplicate_tickers,
            "input_count": len(universe or []),
            "clean_count": len(universe_clean),
        }

    risk_budget = account_equity * risk_per_trade
    score_df["Stop Distance (%)"] = np.maximum(score_df["Stop Loss (%)"], score_df["ATR Context (%)"] * 0.85)
    score_df["Position Size ($)"] = (risk_budget / (score_df["Stop Distance (%)"] / 100)).clip(upper=account_equity / max_positions)
    score_df["Risk per Trade ($)"] = (score_df["Position Size ($)"] * score_df["Stop Distance (%)"] / 100).round(2)
    score_df["Beta Cluster"] = np.where(
        score_df["Market Beta (63D)"].fillna(0) >= 1.2,
        "High Beta",
        np.where(score_df["Market Beta (63D)"].fillna(0) <= 0.8, "Defensive", "Core"),
    )
    score_df["Sector Proxy"] = score_df["Ticker"].str[0]
    score_df = score_df.sort_values(by="AI Score", ascending=False).reset_index(drop=True)
    selected_idx = []
    cluster_counts = {}
    sector_counts = {}
    cluster_cap = 2
    sector_cap = 2
    for idx, r in score_df.iterrows():
        cluster = r["Beta Cluster"]
        sector = r["Sector Proxy"]
        if cluster_counts.get(cluster, 0) >= cluster_cap:
            diagnostics.append({"Ticker": r["Ticker"], "Status": "Rejected", "Reason": f"Correlated exposure cap reached for {cluster}"})
            continue
        if sector_counts.get(sector, 0) >= sector_cap:
            diagnostics.append({"Ticker": r["Ticker"], "Status": "Rejected", "Reason": f"Sector proxy cap reached for {sector}"})
            continue
        selected_idx.append(idx)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_idx) >= max_positions:
            break
    score_df = score_df.loc[selected_idx].copy()
    if score_df.empty:
        diagnostics.append({"Ticker": "*", "Status": "Rejected", "Reason": "No candidates left after exposure caps."})
        return pd.DataFrame(), pd.DataFrame(diagnostics), pd.DataFrame(audit_rows), {
            "invalid_tickers": invalid_tickers,
            "duplicate_tickers": duplicate_tickers,
            "input_count": len(universe or []),
            "clean_count": len(universe_clean),
            "confidence_threshold": min_conf,
        }

    compliance_cols = ["MTF Cheatcode Pass", "Volume Confirmation", "One-Candle Confirmation", "Trend Filter Pass"]
    checklist_hits = score_df[compliance_cols].fillna(False).all(axis=1).sum()
    compliance_pct = float(checklist_hits / max(len(score_df), 1) * 100)
    exception_pct = 100 - compliance_pct
    score_df["Daily Loss Cap (%)"] = max_daily_loss_pct
    score_df["Max Concurrent Positions"] = max_positions
    score_df["Daily Risk Budget ($)"] = round(daily_risk_budget, 2)
    score_df["Projected Risk Used ($)"] = round(float(score_df["Risk per Trade ($)"].sum()), 2)
    score_df["Compliance Pass (%)"] = round(compliance_pct, 1)
    score_df["Exception Rate (%)"] = round(exception_pct, 1)

    drop_cols = [c for c in score_df.columns if c.startswith("_") or c.startswith("z__")]
    score_df = score_df.drop(columns=drop_cols).sort_values(by="AI Score", ascending=False).reset_index(drop=True)

    return score_df, pd.DataFrame(diagnostics), pd.DataFrame(audit_rows), {
        "invalid_tickers": invalid_tickers,
        "duplicate_tickers": duplicate_tickers,
        "input_count": len(universe or []),
        "clean_count": len(universe_clean),
        "confidence_threshold": min_conf,
        "compliance_pct": round(compliance_pct, 1),
        "exception_pct": round(exception_pct, 1),
        "strategy_kill_switch": strategy_kill_switch,
        "kill_switch_reason": kill_switch_reason,
        "projected_daily_risk_used": round(projected_daily_risk_used, 2),
        "daily_risk_budget": round(daily_risk_budget, 2),
        "regime": regime,
    }


@st.cache_data(ttl=180)
def get_ai_stock_selection_bundle(df_history, universe, fundamental_cache, market_shock):
    result = build_ai_stock_selection_table(
        df_history,
        universe,
        fundamental_cache,
        market_shock=market_shock,
    )
    if not isinstance(result, tuple) or len(result) != 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    ai_df, ai_diag_df, ai_audit_df, ai_meta = result
    if not isinstance(ai_df, pd.DataFrame):
        ai_df = pd.DataFrame()
    if not isinstance(ai_diag_df, pd.DataFrame):
        ai_diag_df = pd.DataFrame()
    if not isinstance(ai_audit_df, pd.DataFrame):
        ai_audit_df = pd.DataFrame()
    if not isinstance(ai_meta, dict):
        ai_meta = {}
    return ai_df, ai_diag_df, ai_audit_df, ai_meta


def format_hkt_timestamp(iso_ts: str) -> str:
    """Convert an ISO UTC timestamp string to a Hong Kong time display string."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        hkt = dt.astimezone(ZoneInfo("Asia/Hong_Kong"))
        return hkt.strftime("%Y-%m-%d %H:%M HKT")
    except Exception:
        return "N/A"


def build_high_movement_watchlist(
    df_history,
    universe,
    fundamental_cache,
    market_shock=50,
    min_price=5.0,
    min_dollar_volume=2_000_000,
):
    empty_payload = build_high_movement_payload(
        [],
        market_shock,
        source_inputs=["historical_data", "intraday_5m", "fundamental_cache"],
    )
    if df_history is None or df_history.empty:
        empty_payload["warnings"].append("NO_HISTORICAL_DATA")
        return empty_payload

    candidates = []
    source_rejections = {}
    universe_clean, invalid_tickers, duplicate_tickers = sanitize_universe(universe)
    available = df_history.columns.get_level_values(0).unique()
    intraday_snap = fetch_intraday_5m(list(set(universe_clean) & set(available)))
    qqq_daily = df_history["QQQ"].dropna() if "QQQ" in available else pd.DataFrame()

    for _ticker in invalid_tickers:
        source_rejections["INVALID_TICKER"] = source_rejections.get("INVALID_TICKER", 0) + 1
    for _ticker in duplicate_tickers:
        source_rejections["DUPLICATE_TICKER"] = source_rejections.get("DUPLICATE_TICKER", 0) + 1

    for ticker in universe_clean:
        if ticker not in available:
            source_rejections["MISSING_HISTORY"] = source_rejections.get("MISSING_HISTORY", 0) + 1
            continue

        try:
            df = df_history[ticker].dropna()
            if len(df) < 120:
                source_rejections["INSUFFICIENT_HISTORY"] = source_rejections.get("INSUFFICIENT_HISTORY", 0) + 1
                continue

            price = float(df["Close"].iloc[-1])
            if price < min_price:
                source_rejections["PRICE_BELOW_MIN"] = source_rejections.get("PRICE_BELOW_MIN", 0) + 1
                continue

            avg_dollar_volume = float((df["Close"] * df["Volume"]).tail(20).mean())
            intraday_df = intraday_snap.get(ticker, pd.DataFrame())
            dq = assess_intraday_data_quality(intraday_df)
            if not dq["usable"]:
                source_rejections["INTRADAY_QUALITY_FAILED"] = source_rejections.get("INTRADAY_QUALITY_FAILED", 0) + 1
                continue

            intraday_df = normalize_intraday_bars(intraday_df)
            if intraday_df.empty:
                source_rejections["EMPTY_INTRADAY"] = source_rejections.get("EMPTY_INTRADAY", 0) + 1
                continue

            daily_tail = df.tail(90)
            intraday_plan = compute_intraday_trade_plan(intraday_df, daily_tail, market_shock=market_shock)
            levels = compute_short_term_levels(df)
            shock = compute_ticker_shock(intraday_df, daily_tail)
            st_mom = compute_short_term_momentum(df)
            signal = unified_signal(df)
            structure = classify_structure(signal)
            beta = compute_beta_to_market(df, qqq_daily)
            recent_returns = df["Close"].pct_change().dropna().tail(20).tolist()

            direction = "up"
            if intraday_plan["sell_trigger"] and not intraday_plan["buy_trigger"]:
                direction = "down"
            elif intraday_plan["distance_from_prev_close_pct"] < 0 and not intraday_plan["trend_filter_pass"]:
                direction = "down"

            compression_ready = (
                intraday_plan["cloud_state"] in {"compression/chop", "transition"}
                or intraday_plan["cloud_separation_pct"] <= 0.18
            )
            trigger_proximity = abs(intraday_plan["distance_from_prev_close_pct"]) >= (intraday_plan["trigger_threshold_pct"] * 0.55)
            technical_trigger = bool(
                (compression_ready and trigger_proximity)
                or intraday_plan["one_candle_confirmation"]
                or intraday_plan["buy_trigger"]
                or intraday_plan["sell_trigger"]
            )

            volatility_expansion = float(
                np.clip(
                    np.interp(intraday_plan["cloud_expansion_ratio"], [0.8, 1.0, 1.4], [25, 60, 100]) * 0.55
                    + np.interp(intraday_plan["atr_context_pct"], [0.4, 1.6, 4.0], [20, 65, 100]) * 0.45,
                    0,
                    100,
                )
            )
            volume_anomaly = float(np.clip(np.interp(st_mom["VolAccel"], [0.85, 1.2, 2.2], [15, 60, 100]), 0, 100))
            catalyst_strength = float(
                np.clip(
                    (78 if technical_trigger else 38)
                    + (10 if intraday_plan["volume_confirmation_pass"] else 0)
                    + (6 if abs(shock["shock_z"]) >= 1.0 else 0),
                    0,
                    100,
                )
            )
            order_flow_liquidity = float(
                np.clip(
                    np.interp(avg_dollar_volume, [min_dollar_volume, 15_000_000, 75_000_000], [45, 75, 100]) * 0.65
                    + np.interp(max(0.0, 14.0 - intraday_plan["spread_bps"]), [0.0, 8.0, 14.0], [10, 65, 100]) * 0.35,
                    0,
                    100,
                )
            )
            trend_alignment = float(
                np.clip(
                    intraday_plan["cloud_confidence"] * 0.35
                    + intraday_plan["mtf_cheatcode_score"] * 0.35
                    + (90 if structure in {"Short-Term Breakout 🚀", "Healthy Uptrend 📈"} else 45) * 0.30,
                    0,
                    100,
                )
            )

            recent_low = float(df.tail(5)["Low"].min())
            recent_high = float(df.tail(5)["High"].max())
            entry_anchor = float(levels["Breakout"] if direction == "up" else recent_low)
            zone_half_pct = 0.004
            if direction == "up":
                ideal_entry = {
                    "type": "zone",
                    "min": round(max(price * (1 - zone_half_pct), levels["Pullback_382"]), 2),
                    "max": round(max(entry_anchor, price), 2),
                }
                stop_loss = round(entry_anchor * (1 - max(intraday_plan["stop_loss_pct"], 0.6) / 100), 2)
                profit_target_1 = round(entry_anchor * (1 + max(intraday_plan["target_pct"] * 0.75, 0.9) / 100), 2)
                profit_target_2 = round(entry_anchor * (1 + max(intraday_plan["target_pct"] * 1.20, 1.8) / 100), 2)
            else:
                upper_zone = min(price * (1 + zone_half_pct), recent_high)
                ideal_entry = {
                    "type": "zone",
                    "min": round(min(entry_anchor, upper_zone), 2),
                    "max": round(max(entry_anchor, upper_zone), 2),
                }
                stop_loss = round(entry_anchor * (1 + max(intraday_plan["stop_loss_pct"], 0.6) / 100), 2)
                profit_target_1 = round(entry_anchor * (1 - max(intraday_plan["target_pct"] * 0.75, 0.9) / 100), 2)
                profit_target_2 = round(entry_anchor * (1 - max(intraday_plan["target_pct"] * 1.20, 1.8) / 100), 2)

            catalyst_summary = (
                "Technical compression and trigger proximity within the next 24h."
                if technical_trigger
                else "Reserve candidate: elevated movement factors but catalyst timing is less explicit."
            )
            comparison_note = (
                "Higher 24h move expectancy and shorter holding horizon than Core Top 5."
                if technical_trigger
                else "Reserve high-movement candidate kept separate from Core Top 5 because the trigger is less explicit."
            )
            holding_time = "2-12h" if market_shock >= 45 else "4-24h"

            candidates.append(
                {
                    "asset": ticker,
                    "volatility_expansion": volatility_expansion,
                    "volume_anomaly": volume_anomaly,
                    "catalyst_strength": catalyst_strength,
                    "order_flow_liquidity": order_flow_liquidity,
                    "trend_alignment": trend_alignment,
                    "catalyst_type": "technical",
                    "catalyst_summary": catalyst_summary,
                    "expected_direction": direction,
                    "confidence": intraday_plan["confidence_score"],
                    "ideal_entry": ideal_entry,
                    "stop_loss": stop_loss,
                    "profit_target_1": profit_target_1,
                    "profit_target_2": profit_target_2,
                    "holding_time": holding_time,
                    "rationale": (
                        f"{structure} | cloud {intraday_plan['cloud_state']} | "
                        f"vol accel {st_mom['VolAccel']:.2f}x | trigger {intraday_plan['trigger_threshold_pct']:.2f}%."
                    ),
                    "status": "waiting",
                    "comparison_note": comparison_note,
                    "has_upcoming_catalyst": False,
                    "has_technical_trigger": technical_trigger,
                    "liquidity_ok": bool(avg_dollar_volume >= min_dollar_volume and intraday_plan["spread_bps"] <= 12.0),
                    "spread_bps": intraday_plan["spread_bps"],
                    "_recent_returns": recent_returns,
                    "trace": {
                        "feature_inputs": {
                            "volatility_expansion": round(volatility_expansion, 1),
                            "volume_anomaly": round(volume_anomaly, 1),
                            "catalyst_strength": round(catalyst_strength, 1),
                            "order_flow_liquidity": round(order_flow_liquidity, 1),
                            "trend_alignment": round(trend_alignment, 1),
                        },
                        "source_inputs": ["historical_data", "intraday_5m", "fundamental_cache"],
                        "market_shock": market_shock,
                        "avg_20d_dollar_volume": round(avg_dollar_volume, 0),
                        "spread_bps": intraday_plan["spread_bps"],
                        "beta_63d": round(beta, 2) if np.isfinite(beta) else None,
                    },
                }
            )
        except Exception:
            source_rejections["COMPUTATION_ERROR"] = source_rejections.get("COMPUTATION_ERROR", 0) + 1

    payload = build_high_movement_payload(
        candidates,
        market_shock,
        source_inputs=["historical_data", "intraday_5m", "fundamental_cache"],
    )
    if payload["warnings"] or source_rejections:
        payload["warnings"] = payload["warnings"] + [
            f"SOURCE_REJECTION_COUNTS: {source_rejections}"
        ] if source_rejections else payload["warnings"]
    if not payload["high_movement_top5"] and not payload["warnings"]:
        payload["warnings"].append("NO_QUALIFIED_HIGH_MOVEMENT_CANDIDATES")
    return payload


# =========================================================
# 7b. TRADE TRAP ANALYSIS
# =========================================================

def analyze_trade_trap(ticker, df_history, fundamental_cache, market_shock=50):
    """
    Run a trap / risk analysis on a single ticker the user is considering trading.

    Returns a dict with:
      - valid: bool
      - error: str (if not valid)
      - price / price_change_1d_pct
      - divergence_flags: list of (label, detail) bearish divergence signals
      - macro_flags: list of (label, detail) macro-context warnings
      - reasons_not_to_enter: list[str] (up to 3 primary caution flags)
      - conditions_to_change_view: list[str]
      - signal / structure / regime
      - confidence / momentum_score / shock_score
      - cloud_state / cloud_confidence
      - rsi / vol_accel / failed_breakout / do_not_trade / do_not_trade_reason
      - sentiment_score / sentiment_label
      - beta
    """
    result = {
        "valid": False,
        "error": "",
        "ticker": ticker.upper(),
        "price": None,
        "price_change_1d_pct": None,
        "divergence_flags": [],
        "macro_flags": [],
        "reasons_not_to_enter": [],
        "conditions_to_change_view": [],
        "structure": "N/A",
        "regime": market_regime_from_shock(market_shock),
        "confidence": 0.0,
        "momentum_score": 50.0,
        "shock_score": 0.0,
        "cloud_state": "N/A",
        "cloud_confidence": 0.0,
        "rsi": None,
        "vol_accel": None,
        "failed_breakout": False,
        "do_not_trade": False,
        "do_not_trade_reason": "",
        "sentiment_score": 50,
        "sentiment_label": "N/A",
        "beta": None,
    }

    if df_history.empty:
        result["error"] = "Historical data unavailable."
        return result

    available = df_history.columns.get_level_values(0).unique()
    t = ticker.upper()
    if t not in available:
        result["error"] = f"'{t}' not found in the current data universe. Add it via the sidebar and reload."
        return result

    try:
        df = df_history[t].dropna()
        if len(df) < 30:
            result["error"] = f"Insufficient history for '{t}' (need ≥30 daily bars)."
            return result

        price = float(df["Close"].iloc[-1])
        result["price"] = price
        result["price_change_1d_pct"] = round(
            (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100, 2
        ) if len(df) >= 2 else 0.0

        # --- Price / Volume divergence ---
        st_mom = compute_short_term_momentum(df)
        vol_accel = st_mom["VolAccel"]
        result["vol_accel"] = round(vol_accel, 2)

        divergence_flags = []

        # Rising price but falling volume (classic distribution trap)
        if result["price_change_1d_pct"] is not None and result["price_change_1d_pct"] > 0.5 and vol_accel < 0.9:
            divergence_flags.append((
                "Price↑ / Volume↓ Divergence",
                f"Price is up {result['price_change_1d_pct']:+.1f}% today but volume is only {vol_accel:.2f}× the 20D average. "
                "Rising price on shrinking volume is a classic distribution / retail-trap signal.",
            ))

        # Multi-day price run with decelerating volume
        if st_mom["3D"] > 4 and vol_accel < 1.0:
            divergence_flags.append((
                "3-Day Rally Without Volume Expansion",
                f"Price is up {st_mom['3D']:+.1f}% over 3 days but volume is below average ({vol_accel:.2f}×). "
                "Moves lacking volume conviction are vulnerable to sharp reversals.",
            ))

        # RSI overbought on flat or lower volume
        sig = unified_signal(df)
        rsi = sig["rsi"]
        result["rsi"] = round(rsi, 1)
        if rsi > 72 and vol_accel < 1.1:
            divergence_flags.append((
                "RSI Overbought on Weak Volume",
                f"RSI(14) = {rsi:.1f} (overbought) while volume acceleration is only {vol_accel:.2f}×. "
                "Overbought momentum without institutional volume backing is a late-entry trap.",
            ))

        # Price far above SMA20 (extended move, mean-reversion risk)
        sma20_dist = (price - sig["sma20"]) / sig["sma20"] * 100 if sig["sma20"] > 0 else 0.0
        if sma20_dist > 8:
            divergence_flags.append((
                f"Price Extended {sma20_dist:.1f}% Above 20D SMA",
                f"Trading {sma20_dist:.1f}% above the 20-day moving average. Chasing extended moves increases the risk of "
                "buying into the tail-end of a move when professional sellers are actively distributing.",
            ))

        result["divergence_flags"] = divergence_flags

        # --- Signal / structure ---
        structure = classify_structure(sig)
        result["structure"] = structure

        # --- Intraday / trade plan ---
        qqq_daily = df_history["QQQ"].dropna() if "QQQ" in available else pd.DataFrame()
        intraday_df = fetch_single_intraday_5m(t)
        daily_tail = df.tail(60)

        shock = compute_ticker_shock(intraday_df, daily_tail)
        result["shock_score"] = round(shock["shock_score"], 1)

        plan = compute_intraday_trade_plan(intraday_df, daily_tail, market_shock=market_shock)
        result["confidence"] = round(plan["confidence_score"], 1)
        result["momentum_score"] = round(plan["momentum_score"], 1)
        result["cloud_state"] = plan["cloud_state"]
        result["cloud_confidence"] = round(plan["cloud_confidence"], 1)
        result["failed_breakout"] = plan["failed_breakout"]
        result["do_not_trade"] = plan["do_not_trade"]
        result["do_not_trade_reason"] = plan["do_not_trade_reason"] or ""

        # --- Sentiment ---
        sentiment = calculate_advanced_sentiment(df_history, t)
        result["sentiment_score"] = sentiment.get("score", 50)
        result["sentiment_label"] = sentiment.get("label", "N/A")

        # --- Beta / macro context ---
        beta = compute_beta_to_market(df, qqq_daily) if not qqq_daily.empty else float("nan")
        result["beta"] = round(beta, 2) if np.isfinite(beta) else None

        regime = market_regime_from_shock(market_shock)
        result["regime"] = regime

        macro_flags = []

        # High-shock / stress regime risk
        if market_shock >= 70:
            macro_flags.append((
                f"Adverse Macro Regime ({regime.title()})",
                f"Market Shock Index is {market_shock}/100 — a {regime} environment. "
                "Entering long positions in a stressed macro backdrop dramatically raises the probability of gap-down reversals and stop-outs.",
            ))
        elif market_shock >= 55:
            macro_flags.append((
                f"Elevated Market Stress (Shock={market_shock})",
                f"The market stress index is elevated ({market_shock}/100). "
                "Intraday setups carry higher invalidation risk when macro momentum is against you.",
            ))

        # High beta in a stressed market
        if result["beta"] is not None and result["beta"] > 1.5 and market_shock >= 50:
            macro_flags.append((
                f"High Beta ({result['beta']:.2f}×) in Elevated Stress",
                f"This stock has a 63-day beta of {result['beta']:.2f}× relative to QQQ. "
                "High-beta names amplify market drawdowns — entering during elevated stress is a high-risk proposition.",
            ))

        # Price below SMA200 (against macro trend)
        if price < sig["sma200"] * 0.995:
            sma200_dist = (price - sig["sma200"]) / sig["sma200"] * 100
            macro_flags.append((
                f"Price Below 200D SMA ({sma200_dist:.1f}%)",
                f"Trading {abs(sma200_dist):.1f}% below the 200-day moving average. "
                "The macro trend structure is bearish / broken — buying against the dominant trend is typically a low-probability trade.",
            ))

        # Price below SMA50 (intermediate trend broken)
        if price < sig["sma50"] * 0.995 and price >= sig["sma200"] * 0.995:
            sma50_dist = (price - sig["sma50"]) / sig["sma50"] * 100
            macro_flags.append((
                f"Intermediate Trend Broken (Price {sma50_dist:.1f}% vs 50D SMA)",
                f"Price is {abs(sma50_dist):.1f}% below its 50-day moving average. "
                "Intermediate trend structure is broken — dip-buying in a broken trend often leads to catching a falling knife.",
            ))

        result["macro_flags"] = macro_flags

        # --- Compile top 3 reasons not to enter ---
        reasons = []

        # Priority 1: engine do-not-trade
        if plan["do_not_trade"] and plan["do_not_trade_reason"]:
            reasons.append(f"AI Engine veto: {plan['do_not_trade_reason']}")

        # Priority 2: failed breakout
        if plan["failed_breakout"]:
            reasons.append("Failed breakout detected — the attempted move was rejected and could accelerate to the downside.")

        # Priority 3: divergence flags (most severe first)
        for label, _ in divergence_flags:
            if len(reasons) >= 3:
                break
            reasons.append(f"Divergence — {label}")

        # Priority 4: macro flags
        for label, _ in macro_flags:
            if len(reasons) >= 3:
                break
            reasons.append(f"Macro context — {label}")

        # Priority 5: low confidence
        if len(reasons) < 3 and plan["confidence_score"] < 40:
            reasons.append(f"Low intraday confidence score ({plan['confidence_score']:.0f}/100) — setup conviction is insufficient to justify risk.")

        # Priority 6: cloud misalignment
        if len(reasons) < 3 and plan["cloud_state"] not in {"bullish trend", "transition"}:
            reasons.append(f"Cloud misalignment — current cloud state is '{plan['cloud_state']}', indicating no sustained bullish structure.")

        # Priority 7: regime mismatch for long
        if len(reasons) < 3 and regime in {"stress", "shock"}:
            reasons.append(f"Regime mismatch — a {regime} regime strongly favors cash or short exposure over new long entries.")

        # Fill to 3 if still short
        while len(reasons) < 3:
            reasons.append("Setup does not meet minimum multi-timeframe alignment thresholds.")

        result["reasons_not_to_enter"] = reasons[:3]

        # --- Conditions to change view ---
        conditions = []

        if divergence_flags:
            conditions.append(
                f"Volume expands to ≥1.3× the 20D average on a green close, confirming real demand rather than distribution."
            )
        if rsi > 68:
            conditions.append(
                f"RSI resets below 60 via a constructive pullback (not a crash), then re-bases above the 20D SMA with volume support."
            )
        if plan["cloud_state"] not in {"bullish trend"}:
            conditions.append(
                f"Intraday Ripster Cloud transitions to 'bullish trend' state with cloud confidence ≥65 — indicating structured buyers returning."
            )
        if not plan["mtf_cheatcode_pass"]:
            conditions.append(
                "Multi-timeframe cheatcode aligns (5m, daily, and weekly cloud structures all bullish simultaneously)."
            )
        if macro_flags:
            conditions.append(
                f"Market Shock Index drops below 45 (current: {market_shock}) and price reclaims its 50D SMA on above-average volume."
            )
        if plan["failed_breakout"]:
            conditions.append(
                "Price reclaims the breakout level cleanly on a volume surge ≥1.5× average, invalidating the failed-breakout pattern."
            )
        if not conditions:
            conditions.append("All primary signals align: cloud bullish, MTF pass, volume ≥1.2× average, RSI 50–65, and Market Shock <45.")
        if len(conditions) < 3:
            conditions.append(
                f"Sentiment score improves above 65 (current: {result['sentiment_score']}) with confirmed positive technical structure."
            )

        result["conditions_to_change_view"] = conditions[:4]
        result["valid"] = True

    except Exception as e:
        result["error"] = f"Analysis error: {e}"

    return result


# =========================================================
# 8. USER INTERFACE
# =========================================================

st.title("📈 Wealth Terminal v12.0")
universe = get_base_universe()

# --- Market regime banner ---
with st.spinner("Syncing intraday market stress regime..."):
    intraday_index = fetch_single_intraday_5m("QQQ", days=3)
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

# --- Sidebar universe controls ---
st.sidebar.markdown("### ➕ Add Custom Stocks")

manual_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):",
    placeholder="e.g., TSLA, AAPL, PLTR",
)

manual_list = []
if manual_input:
    manual_list = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

custom_select = st.sidebar.multiselect(
    "Or select from universe:",
    options=universe,
    default=[],
)

user_added_tickers = list(set(manual_list + custom_select))
full_universe_raw = universe + user_added_tickers
full_universe, invalid_tickers_ui, duplicate_tickers_ui = sanitize_universe(full_universe_raw)

st.sidebar.success(f"Tracking {len(full_universe)} valid tickers")
if invalid_tickers_ui:
    st.sidebar.warning(f"Invalid tickers ignored: {', '.join(invalid_tickers_ui[:8])}")
if duplicate_tickers_ui:
    st.sidebar.info(f"Duplicate tickers ignored: {', '.join(duplicate_tickers_ui[:8])}")

# --- Data loads ---
with st.spinner("Syncing technical historical structures..."):
    historical_data = fetch_historical_data(full_universe)

with st.spinner("Extracting corporate fundamental structures..."):
    fundamental_cache = fetch_fundamental_metrics(full_universe)

with st.spinner("Checking intraday feed quality..."):
    health_universe = full_universe[: min(12, len(full_universe))]
    intraday_health_snap = fetch_intraday_5m(health_universe)
    health_rows = []
    for t in health_universe:
        q = assess_intraday_data_quality(intraday_health_snap.get(t, pd.DataFrame()))
        health_rows.append(
            {
                "Ticker": t,
                "Usable": q["usable"],
                "Fresh": q["fresh"],
                "Stale (min)": q["stale_minutes"],
                "Missing Bars (%)": round(q["missing_ratio"] * 100, 1),
                "Warnings": " | ".join(q["warnings"]),
            }
        )
    intraday_health_df = pd.DataFrame(health_rows)

if not intraday_health_df.empty and (~intraday_health_df["Usable"]).any():
    bad_count = int((~intraday_health_df["Usable"]).sum())
    st.warning(f"Intraday data quality warning: {bad_count} sampled tickers have stale or missing bars. Review AI diagnostics before trading.")

if not historical_data.empty:
    ai_df_shared, ai_diag_df_shared, ai_audit_df_shared, ai_meta_shared = get_ai_stock_selection_bundle(
        historical_data,
        full_universe,
        fundamental_cache,
        market_shock,
    )
    macro_df_shared = calculate_macro_trends(historical_data, full_universe, fundamental_cache)
else:
    ai_df_shared, ai_diag_df_shared, ai_audit_df_shared, ai_meta_shared = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    macro_df_shared = pd.DataFrame()

# --- Tabs ---
(
    tab_momentum,
    tab_breakout,
    tab_pullback,
    tab_sentiment,
    tab_macro,
    tab_ai,
    tab_top5,
    tab_high_movement,
    tab_trap,
) = st.tabs(
    [
        "⚡ Short-Term Momentum",
        "🚀 Breakout Radar",
        "📉 Pullback Scanner",
        "🔮 Technical Sentiment",
        "🏛️ Macro Wealth & Long-Term Investment",
        "🤖 AI Stock Selection Engine",
        "🏆 Top 5 Today",
        "Top 5 High-Movement (Next 24h)",
        "🪤 Trade Trap Checker",
    ]
)

# =========================================================
# TAB 1: SHORT-TERM MOMENTUM
# =========================================================

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

# =========================================================
# TAB 2: BREAKOUT RADAR
# =========================================================

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

# =========================================================
# TAB 3: PULLBACK SCANNER
# =========================================================

with tab_pullback:
    st.subheader("📉 Pullback Scanner — 38.2% to 61.8% Retracement Zones")

    if not historical_data.empty:
        df_pullback = pullback_scanner(historical_data, full_universe)

        if not df_pullback.empty:

            # Auto-highlighting logic
            def highlight_pullback(row):
                price = row["Price"]
                pb382 = row["Pullback 38.2%"]
                pb618 = row["Pullback 61.8%"]

                # Strong trend pullback (near 38.2%)
                if abs(price - pb382) <= abs(pb618 - pb382) * 0.25:
                    return ["background-color: #14532d; color: white"] * len(row)

                # Deep dip reversal (near 61.8%)
                if abs(price - pb618) <= abs(pb618 - pb382) * 0.25:
                    return ["background-color: #1e3a8a; color: white"] * len(row)

                # Trend may be failing (below 61.8%)
                if price < pb618:
                    return ["background-color: #7f1d1d; color: white"] * len(row)

                # Neutral zone
                return [""] * len(row)

            st.dataframe(
                df_pullback.style.apply(highlight_pullback, axis=1),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No assets currently in optimal pullback zones.")
    else:
        st.error("Historical data unavailable.")


# =========================================================
# TAB 4: TECHNICAL SENTIMENT (REGIME-AWARE)
# =========================================================

with tab_sentiment:
    st.subheader("Dynamic Fear & Greed Structural Proxies")

    if historical_data.empty:
        st.error("Historical data unavailable.")
    else:
        selected_ticker = st.selectbox("Select Target Engine Asset:", full_universe)

        if selected_ticker in historical_data.columns.get_level_values(0):
            sentiment = calculate_advanced_sentiment(historical_data, selected_ticker)

            if sentiment["status"] == "Active":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Aggregate Score", sentiment["score"], sentiment["label"])
                with col2:
                    st.metric("RSI (14 Daily)", sentiment["metrics"]["rsi_14"])
                with col3:
                    st.metric("Volatility Multiplier", f"{sentiment['metrics']['volatility_ratio']}x")
                with col4:
                    st.metric("Ripster MTF Score", sentiment["metrics"]["ripster_mtf_score"])

                st.markdown("### ☁️ Ripster Multi-Timeframe Read")
                rip_col1, rip_col2, rip_col3, rip_col4 = st.columns(4)
                with rip_col1:
                    st.metric("Alignment Score", sentiment["metrics"]["ripster_alignment_score"])
                with rip_col2:
                    st.metric("Daily Cloud", sentiment["metrics"]["ripster_daily_state"])
                with rip_col3:
                    st.metric("Weekly Cloud", sentiment["metrics"]["ripster_weekly_state"])
                with rip_col4:
                    st.metric(
                        "Cheatcode Score",
                        sentiment["metrics"]["ripster_cheatcode_score"],
                        "PASS" if sentiment["metrics"]["ripster_cheatcode_pass"] else "WATCH",
                    )

                ripster_components = pd.DataFrame(
                    [
                        {"Component": "Daily Cloud Confidence", "Value": sentiment["metrics"]["ripster_daily_confidence"]},
                        {"Component": "Weekly Cloud Confidence", "Value": sentiment["metrics"]["ripster_weekly_confidence"]},
                        {"Component": "Weekly Return (%)", "Value": sentiment["metrics"]["ripster_weekly_return_pct"]},
                        {"Component": "Trend Confirmation", "Value": sentiment["metrics"]["ripster_trend_score"]},
                        {"Component": "Momentum Confirmation", "Value": sentiment["metrics"]["ripster_momentum_score"]},
                        {"Component": "Volatility Context", "Value": sentiment["metrics"]["ripster_volatility_score"]},
                        {
                            "Component": f"Structure ({sentiment['metrics']['ripster_structure_label']})",
                            "Value": sentiment["metrics"]["ripster_structure_score"],
                        },
                    ]
                )
                st.dataframe(ripster_components, use_container_width=True, hide_index=True)

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
                    np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
                )
                atr5 = tr.rolling(5).mean()
                atr20 = tr.rolling(20).mean()
                vol_ratio_series = atr5 / atr20

                # Price + signals
                fig_price = go.Figure()
                fig_price.add_trace(
                    go.Scatter(
                        x=close.index,
                        y=close,
                        name="Close",
                        line=dict(color="#38bdf8", width=2),
                    )
                )
                fig_price.add_trace(
                    go.Scatter(
                        x=sma20.index,
                        y=sma20,
                        name="SMA20",
                        line=dict(color="#f59e0b", dash="dash"),
                    )
                )

                buy_signals = []
                sell_signals = []

                for i in range(1, len(close)):
                    if (
                        close.iloc[i] > sma20.iloc[i]
                        and close.iloc[i - 1] <= sma20.iloc[i - 1]
                        and rsi_series.iloc[i] > 50
                        and vol_ratio_series.iloc[i] > 1.0
                    ):
                        buy_signals.append((close.index[i], close.iloc[i]))

                    if (
                        close.iloc[i] < sma20.iloc[i]
                        and close.iloc[i - 1] < sma20.iloc[i - 1]
                        and rsi_series.iloc[i] < 40
                    ):
                        sell_signals.append((close.index[i], close.iloc[i]))

                for t, p in buy_signals:
                    fig_price.add_annotation(
                        x=t,
                        y=p,
                        text="⬆ BUY",
                        showarrow=True,
                        arrowhead=1,
                        font=dict(color="#22c55e"),
                    )

                for t, p in sell_signals:
                    fig_price.add_annotation(
                        x=t,
                        y=p,
                        text="⬇ SELL",
                        showarrow=True,
                        arrowhead=1,
                        font=dict(color="#ef4444"),
                    )

                fig_price.update_layout(
                    title=f"{selected_ticker} — Price with Signals",
                    template="plotly_dark",
                    height=320,
                )
                st.plotly_chart(fig_price, use_container_width=True)

                # RSI / Vol / Price views
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=rsi_series.index,
                        y=rsi_series,
                        mode="lines",
                        name="RSI 14",
                        line=dict(color="#38bdf8", width=2),
                    )
                )
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.15, line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.15, line_width=0)
                fig_rsi.update_layout(
                    title=f"{selected_ticker} — RSI (14)",
                    template="plotly_dark",
                    height=230,
                )

                fig_price2 = go.Figure()
                fig_price2.add_trace(
                    go.Scatter(
                        x=close.index,
                        y=close,
                        name="Close",
                        line=dict(color="#38bdf8", width=2),
                    )
                )
                fig_price2.add_trace(
                    go.Scatter(
                        x=sma20.index,
                        y=sma20,
                        name="SMA20",
                        line=dict(color="#f59e0b", dash="dash"),
                    )
                )
                fig_price2.update_layout(
                    title=f"{selected_ticker} — Price vs SMA20",
                    template="plotly_dark",
                    height=260,
                )

                fig_vol = go.Figure()
                fig_vol.add_trace(
                    go.Scatter(
                        x=vol_ratio_series.index,
                        y=vol_ratio_series,
                        name="ATR5 / ATR20",
                        line=dict(color="#ef4444", width=2),
                    )
                )
                fig_vol.update_layout(
                    title=f"{selected_ticker} — Volatility Ratio",
                    template="plotly_dark",
                    height=230,
                )

                st.plotly_chart(fig_price2, use_container_width=True)
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.plotly_chart(fig_vol, use_container_width=True)

                # Simple backtest (10–30 day swing)
                st.markdown("### 📈 Backtest Results (10–30 Day Swing Strategy)")

                returns = []
                trade_lengths = []
                position = None
                entry_price = None
                entry_index = None

                for i in range(1, len(close)):
                    if (
                        position is None
                        and close.iloc[i] > sma20.iloc[i]
                        and rsi_series.iloc[i] > 50
                        and vol_ratio_series.iloc[i] > 1.0
                    ):
                        position = "LONG"
                        entry_price = close.iloc[i]
                        entry_index = i

                    elif position == "LONG" and (
                        (
                            close.iloc[i] < sma20.iloc[i]
                            and close.iloc[i - 1] < sma20.iloc[i - 1]
                            and rsi_series.iloc[i] < 40
                        )
                        or vol_ratio_series.iloc[i] < 0.8
                    ):
                        ret = (close.iloc[i] - entry_price) / entry_price
                        returns.append(ret)
                        if entry_index is not None:
                            trade_lengths.append(i - entry_index)
                        position = None

                if returns:
                    win_rate = 100 * sum(r > 0 for r in returns) / len(returns)
                    avg_return = 100 * np.mean(returns)
                    avg_len = np.mean(trade_lengths) if trade_lengths else 0
                else:
                    win_rate = 0
                    avg_return = 0
                    avg_len = 0

                col_bt1, col_bt2, col_bt3 = st.columns(3)
                with col_bt1:
                    st.metric("Win Rate (%)", f"{win_rate:.1f}")
                with col_bt2:
                    st.metric("Avg Trade Return (%)", f"{avg_return:.2f}")
                with col_bt3:
                    st.metric("Avg Holding (bars)", f"{avg_len:.1f}")

                wf = run_walk_forward_validation(ticker_df)
                st.markdown("### 🧪 Walk-Forward OOS Validation (Regime-Aware)")
                if wf.get("status") == "ok":
                    overall = wf["overall"]
                    col_wf1, col_wf2, col_wf3, col_wf4, col_wf5, col_wf6 = st.columns(6)
                    with col_wf1:
                        st.metric("OOS Trades", overall["trades"])
                    with col_wf2:
                        st.metric("OOS Win Rate (%)", f"{overall['win_rate']:.1f}")
                    with col_wf3:
                        st.metric("Expectancy (%)", f"{overall['expectancy_pct']:.3f}")
                    with col_wf4:
                        st.metric("Max Drawdown (%)", f"{overall['max_drawdown_pct']:.2f}")
                    with col_wf5:
                        st.metric("Sharpe / Sortino", f"{overall['sharpe']:.2f} / {overall['sortino']:.2f}")
                    with col_wf6:
                        st.metric("Profit Factor", f"{overall['profit_factor']:.2f}")

                    regime_df = wf.get("regime_df", pd.DataFrame())
                    if not regime_df.empty:
                        st.markdown("#### OOS Performance by Regime")
                        st.dataframe(regime_df, use_container_width=True, hide_index=True)

                    cloud_alignment_df = wf.get("cloud_alignment_df", pd.DataFrame())
                    if not cloud_alignment_df.empty:
                        st.markdown("#### Cloud-Aligned vs Non-Aligned (OOS)")
                        st.dataframe(cloud_alignment_df, use_container_width=True, hide_index=True)

                    ablation_df = wf.get("ablation_df", pd.DataFrame())
                    if not ablation_df.empty:
                        st.markdown("#### Feature Ablation (OOS)")
                        st.dataframe(ablation_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Insufficient history for walk-forward validation.")

                trend_phase = classify_structure(unified_signal(ticker_df))
                signal_quality, narrative_lines, score_components = compute_signal_quality_and_narrative(
                    close,
                    sma20,
                    rsi_series,
                    vol_ratio_series,
                    returns,
                    win_rate,
                    avg_return,
                    trend_phase,
                )

                # Ticker shock for regime-aware narrative
                intraday_df_single = fetch_single_intraday_5m(selected_ticker, days=3)
                ticker_shock_obj = compute_ticker_shock(intraday_df_single, ticker_df.tail(30))
                ticker_shock_score = ticker_shock_obj["shock_score"]

                st.markdown("### 🧠 Signal Quality & Regime-Aware Narrative")
                col_sq1, col_sq2, col_sq3, col_sq4, col_sq5 = st.columns(5)
                with col_sq1:
                    st.metric("Signal Quality", signal_quality)
                with col_sq2:
                    st.metric("Trend Score", score_components["trend_score"])
                with col_sq3:
                    st.metric("Momentum Score", score_components["momentum_score"])
                with col_sq4:
                    st.metric("Volatility Score", score_components["vol_score"])
                with col_sq5:
                    st.metric("Structure Score", score_components["structure_score"])

                regime_lines = build_regime_aware_narrative(
                    market_shock,
                    ticker_shock_score,
                    trend_phase,
                    sentiment["label"],
                    signal_quality,
                )

                st.markdown("#### Regime-Aware Narrative")
                for line in regime_lines:
                    st.markdown(f"- {line}")

            else:
                st.error("Sentiment engine returned an error state.")
        else:
            st.warning("Selected ticker not found in historical data universe.")

# =========================================================
# TAB 5: MACRO WEALTH & LONG-TERM INVESTMENT
# =========================================================

with tab_macro:
    st.subheader("🏛️ Macro Wealth & Long-Term Investment")
    if not historical_data.empty:
        macro_df = macro_df_shared
        if not macro_df.empty:
            st.dataframe(macro_df, use_container_width=True, hide_index=True)
        else:
            st.info("No macro structures available for current universe (insufficient history).")
    else:
        st.error("Historical data unavailable.")

# =========================================================
# TAB 6: AI STOCK SELECTION ENGINE
# =========================================================

with tab_ai:
    st.subheader("🤖 AI Stock Selection Engine — Short-Term Tactical Focus")
    if not historical_data.empty:
        ai_df, ai_diag_df, ai_audit_df, ai_meta = (
            ai_df_shared,
            ai_diag_df_shared,
            ai_audit_df_shared,
            ai_meta_shared,
        )
        st.caption(
            f"Regime confidence threshold: {ai_meta.get('confidence_threshold', 'N/A')} | "
            f"Input tickers: {ai_meta.get('input_count', 0)} | "
            f"Valid tickers: {ai_meta.get('clean_count', 0)} | "
            f"Checklist compliance: {ai_meta.get('compliance_pct', 0)}% | "
            f"Exceptions: {ai_meta.get('exception_pct', 0)}%"
        )
        if ai_meta.get("strategy_kill_switch"):
            st.error(f"Strategy kill-switch active: {ai_meta.get('kill_switch_reason', 'Risk control')}")
        if not ai_df.empty:
            st.dataframe(ai_df, use_container_width=True, hide_index=True)
        else:
            st.info("AI engine did not find any qualified candidates (check data coverage and universe).")

        if not intraday_health_df.empty:
            with st.expander("Intraday Feed Quality Sample", expanded=False):
                st.dataframe(intraday_health_df, use_container_width=True, hide_index=True)

        if not ai_diag_df.empty:
            with st.expander("Rejected / Filtered Symbols Diagnostics", expanded=False):
                st.dataframe(ai_diag_df, use_container_width=True, hide_index=True)

        if not ai_audit_df.empty:
            with st.expander("Signal Audit Log", expanded=False):
                st.dataframe(ai_audit_df, use_container_width=True, hide_index=True)
    else:
        st.error("Historical data unavailable.")

# =========================================================
# TAB 7: TOP 5 TODAY
# =========================================================

with tab_top5:
    st.subheader("🏆 Top 5 Trading Day Candidates")
    st.caption("Ranked by AI Score from the AI Stock Selection Engine. Only qualified, regime-aware candidates are shown.")
    if not historical_data.empty:
        top5_ai_df, top5_meta = ai_df_shared, ai_meta_shared
        if top5_meta.get("strategy_kill_switch"):
            st.error(f"Strategy kill-switch active: {top5_meta.get('kill_switch_reason', 'Risk control')} — no candidates today.")
        elif top5_ai_df.empty:
            st.info("The AI engine found no qualified candidates for today. Check data coverage, market regime, or universe settings.")
        else:
            _top5_cols_preferred = [
                "Ticker",
                "AI Score",
                "Confidence Score",
                "Net Edge (bps)",
                "Reward/Risk",
                "Setup Grade",
                "Regime",
                "Intraday Momentum Score",
                "Cloud Confidence",
                "MTF Cheatcode Pass",
                "Volume Confirmation",
            ]
            _top5_cols_available = [c for c in _top5_cols_preferred if c in top5_ai_df.columns]
            if not _top5_cols_available:
                st.info("Top 5 columns unavailable in current AI output format. Showing first 5 rows instead.")
                top5_display = top5_ai_df.head(5).reset_index(drop=True)
            else:
                top5_display = top5_ai_df.head(5)[_top5_cols_available].reset_index(drop=True)
            top5_display.index = top5_display.index + 1
            st.dataframe(top5_display, use_container_width=True)
            st.caption(
                f"Regime confidence threshold: {top5_meta.get('confidence_threshold', 'N/A')} | "
                f"Input tickers: {top5_meta.get('input_count', 0)} | "
                f"Valid tickers: {top5_meta.get('clean_count', 0)}"
            )
    else:
        st.error("Historical data unavailable — cannot compute top 5 candidates.")

# =========================================================
# TAB 8: TOP 5 HIGH-MOVEMENT (NEXT 24H)
# =========================================================

with tab_high_movement:
    st.subheader("Top 5 High-Movement (Next 24h)")
    st.caption(
        "Independent watchlist ranked for next-24h movement probability. Hard risk filters stay intact; "
        "reserve backfill is only used when needed to maintain five names."
    )
    if not historical_data.empty:
        high_movement_payload = build_high_movement_watchlist(
            historical_data,
            full_universe,
            fundamental_cache,
            market_shock=market_shock,
        )
        comparison_col, watchlist_col = st.columns([1, 1.35], gap="large")

        with comparison_col:
            st.markdown("#### Top 5 (Core Picks)")
            comparison_top5_df, comparison_top5_meta = ai_df_shared, ai_meta_shared
            if comparison_top5_meta.get("strategy_kill_switch"):
                st.error(f"Strategy kill-switch active: {comparison_top5_meta.get('kill_switch_reason', 'Risk control')}")
            elif comparison_top5_df.empty:
                st.info("No Core Top 5 candidates available for comparison.")
            else:
                comparison_cols = [
                    "Ticker",
                    "AI Score",
                    "Confidence Score",
                    "Reward/Risk",
                    "Setup Grade",
                    "Regime",
                ]
                available_comparison_cols = [c for c in comparison_cols if c in comparison_top5_df.columns]
                if not available_comparison_cols:
                    st.info("Core Top 5 columns unavailable in current AI output format. Showing first 5 rows instead.")
                    st.dataframe(
                        comparison_top5_df.head(5).reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(
                        comparison_top5_df.head(5)[available_comparison_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )

        with watchlist_col:
            regime = high_movement_payload.get("regime", {})
            st.caption(
                f"Generated at: {format_hkt_timestamp(high_movement_payload.get('generated_at', ''))} | "
                f"Regime: {regime.get('state', 'N/A')} | {regime.get('summary', '')}"
            )
            for idx, candidate in enumerate(high_movement_payload.get("high_movement_top5", []), start=1):
                display_ctx = build_candidate_display_context(candidate)
                status_meta = display_ctx["status_badge"]
                direction_label = candidate["expected_direction"].upper()
                entry = candidate["ideal_entry"]
                if isinstance(entry, dict):
                    entry_label = f"{entry.get('min', 'N/A')} - {entry.get('max', 'N/A')}"
                else:
                    entry_label = entry

                st.markdown(
                    f"""
                    <div style="border:1px solid #334155; border-left:6px solid {display_ctx['direction_color']};
                    padding:0.9rem 1rem; border-radius:10px; margin-bottom:0.8rem; background:#0f172a;">
                        <div style="display:flex; justify-content:space-between; gap:1rem; align-items:center;">
                            <div>
                                <div style="font-size:1.05rem; font-weight:700;">{idx}. {candidate['asset']}</div>
                                <div style="color:#94a3b8;">Score {candidate['score']} | Confidence {candidate['confidence']}/100 ({display_ctx['confidence_band']})</div>
                            </div>
                            <div style="display:flex; gap:0.5rem; align-items:center;">
                                <span style="color:{display_ctx['direction_color']}; font-weight:700;">{direction_label}</span>
                                <span style="background:{status_meta['bg']}; color:{status_meta['fg']}; padding:0.2rem 0.6rem; border-radius:999px; font-size:0.8rem;">{status_meta['label']}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                metric_cols = st.columns(5)
                metric_cols[0].metric("Entry Zone", entry_label)
                metric_cols[1].metric("Stop Loss", candidate["stop_loss"])
                metric_cols[2].metric("Profit Target 1", candidate["profit_target_1"])
                metric_cols[3].metric("Profit Target 2", candidate["profit_target_2"])
                metric_cols[4].metric("R/R", candidate["risk_reward_ratio"])
                st.write(
                    f"**Catalyst ({candidate['catalyst_type']}):** {candidate['catalyst_summary']}  \n"
                    f"**Holding Time:** {candidate['holding_time']}  \n"
                    f"**Rationale:** {candidate['rationale']}  \n"
                    f"**Comparison Note:** {candidate['comparison_note']}"
                )
                breakdown = candidate.get("score_breakdown", {})
                st.caption(
                    "Score breakdown — "
                    f"Volatility: {breakdown.get('volatility_expansion', 0)} | "
                    f"Volume: {breakdown.get('volume_anomaly', 0)} | "
                    f"Catalyst: {breakdown.get('catalyst_strength', 0)} | "
                    f"Order flow/liquidity: {breakdown.get('order_flow_liquidity', 0)} | "
                    f"Trend: {breakdown.get('trend_alignment', 0)}"
                )
                with st.expander(f"Trace metadata — {candidate['asset']}", expanded=False):
                    st.json(candidate.get("trace", {}))

            if high_movement_payload.get("warnings"):
                for warning in high_movement_payload["warnings"]:
                    st.warning(warning)

            with st.expander("Watchlist filters & trace", expanded=False):
                st.json(
                    {
                        "filters": high_movement_payload.get("filters", {}),
                        "trace_metadata": high_movement_payload.get("trace_metadata", {}),
                    }
                )
    else:
        st.error("Historical data unavailable — cannot compute the high-movement watchlist.")

# =========================================================
# TAB 9: TRADE TRAP CHECKER
# =========================================================

with tab_trap:
    st.subheader("🪤 Trade Trap Checker")
    st.caption(
        "Type any ticker you are considering trading. The engine will analyse whether it looks like a trap for smaller investors, "
        "flag price/volume divergences, check macro context, and give you 3 concrete reasons to stay out — plus what would need to change for the view to flip."
    )

    trap_ticker_input = st.text_input(
        "Enter a ticker to analyse:",
        placeholder="e.g. TSLA",
        max_chars=10,
        key="trap_ticker_input",
    ).strip().upper()

    if trap_ticker_input:
        with st.spinner(f"Running trap analysis on {trap_ticker_input}…"):
            trap_result = analyze_trade_trap(
                trap_ticker_input,
                historical_data,
                fundamental_cache,
                market_shock=market_shock,
            )

        if not trap_result["valid"]:
            st.error(trap_result["error"])
        else:
            # ── Header metrics row ──────────────────────────────────────────────
            st.markdown(f"### Analysis for **{trap_result['ticker']}**")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Price", f"${trap_result['price']:,.2f}" if trap_result["price"] else "N/A",
                          delta=f"{trap_result['price_change_1d_pct']:+.2f}% today" if trap_result["price_change_1d_pct"] is not None else None)
            with m2:
                st.metric("Confidence Score", f"{trap_result['confidence']:.0f}/100")
            with m3:
                st.metric("Intraday Momentum", f"{trap_result['momentum_score']:.0f}/100")
            with m4:
                st.metric("Cloud State", trap_result["cloud_state"])
            with m5:
                st.metric("Regime", trap_result["regime"].title())

            m6, m7, m8, m9 = st.columns(4)
            with m6:
                st.metric("RSI (14D)", f"{trap_result['rsi']:.1f}" if trap_result["rsi"] is not None else "N/A")
            with m7:
                st.metric("Volume Accel", f"{trap_result['vol_accel']:.2f}×" if trap_result["vol_accel"] is not None else "N/A")
            with m8:
                st.metric("Sentiment", f"{trap_result['sentiment_score']:.0f} — {trap_result['sentiment_label']}")
            with m9:
                beta_str = f"{trap_result['beta']:.2f}×" if trap_result["beta"] is not None else "N/A"
                st.metric("Beta (63D vs QQQ)", beta_str)

            if trap_result["do_not_trade"]:
                st.error(f"⛔ AI Engine Veto: {trap_result['do_not_trade_reason']}")
            if trap_result["failed_breakout"]:
                st.warning("⚠️ Failed Breakout Detected — the attempted breakout was rejected.")

            st.divider()

            # ── Price / Volume Divergence ────────────────────────────────────────
            st.markdown("#### 📊 Price / Volume Divergence Signals")
            if trap_result["divergence_flags"]:
                for label, detail in trap_result["divergence_flags"]:
                    with st.expander(f"🔴 {label}", expanded=True):
                        st.write(detail)
            else:
                st.success("✅ No significant price/volume divergence detected at this time.")

            st.divider()

            # ── Macro Context ────────────────────────────────────────────────────
            st.markdown("#### 🏛️ Macro Context")
            if trap_result["macro_flags"]:
                for label, detail in trap_result["macro_flags"]:
                    with st.expander(f"🟠 {label}", expanded=True):
                        st.write(detail)
            else:
                st.success("✅ No major macro headwinds identified for this trade.")

            st.divider()

            # ── 3 Reasons Not To Enter ───────────────────────────────────────────
            st.markdown("#### 🚫 3 Reasons Not to Enter This Trade")
            for i, reason in enumerate(trap_result["reasons_not_to_enter"], 1):
                st.markdown(f"**{i}.** {reason}")

            st.divider()

            # ── What Would Need to Change ────────────────────────────────────────
            st.markdown("#### ✅ What Would Need to Be Satisfied to Change This View")
            for condition in trap_result["conditions_to_change_view"]:
                st.markdown(f"- {condition}")
    else:
        st.info("Enter a ticker above to run the trap analysis.")
