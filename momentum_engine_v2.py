# momentum_engine_v2.py
# Leopold-inspired momentum swing engine v2
# Dependencies: pandas, numpy

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


# ==========
# CONFIG
# ==========

@dataclass
class EngineConfig:
    risk_per_trade: float = 0.01        # 1% of equity
    min_mqs: int = 70                   # Minimum Momentum Quality Score
    min_avg_volume: int = 1_500_000
    max_spread_pct: float = 0.15        # 0.15%
    max_overnight_gap_pct: float = 2.0
    max_atr_pct: float = 6.0
    max_intraday_atr_pct: float = 4.5
    vix_threshold: float = 20.0         # pass in via context if needed
    earnings_buffer_days: int = 5


# ==========
# INTERNAL NORMALIZERS (robustness only; no strategy change)
# ==========

def _as_1d_series(x, name: str = "value") -> pd.Series:
    """
    Normalize input into a 1D numeric Series.
    Handles Series, single-col DataFrame, ndarray (n,1)/(1,n), lists.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float, name=name)
        if name in x.columns:
            c = x[name]
            s = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        else:
            s = x.iloc[:, 0]
    elif isinstance(x, np.ndarray):
        s = pd.Series(np.ravel(x), name=name)
    else:
        s = pd.Series(x, name=name)

    s = pd.to_numeric(s, errors="coerce")
    return s


def _series_col(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Safely extract a 1D Series column from a DataFrame.
    Handles duplicated column names and MultiIndex side-effects where df[col] returns DataFrame.
    """
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float, name=col)

    val = df[col]
    if isinstance(val, pd.DataFrame):
        if val.shape[1] == 0:
            return pd.Series(index=df.index, dtype=float, name=col)
        val = val.iloc[:, 0]
    return _as_1d_series(val, name=col).reindex(df.index)


# ==========
# INDICATORS
# ==========

def ema(series: pd.Series, span: int) -> pd.Series:
    s = _as_1d_series(series, name=getattr(series, "name", "value"))
    return s.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    s = _as_1d_series(series, name=getattr(series, "name", "value"))
    return s.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _as_1d_series(series, name="Close")
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _series_col(df, "High")
    low = _series_col(df, "Low")
    close = _series_col(df, "Close")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _as_1d_series(series, name="Close")
    fast_ema = ema(s, fast)
    slow_ema = ema(s, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    # anchor_idx: integer index where swing low occurs
    px = _series_col(df, "Close").copy()
    vol = _series_col(df, "Volume").copy()
    px.iloc[:anchor_idx] = np.nan
    vol.iloc[:anchor_idx] = np.nan
    cum_pv = (px * vol).cumsum()
    cum_v = vol.cumsum()
    vwap = cum_pv / (cum_v + 1e-9)
    return vwap


# ==========
# STRUCTURAL SCORES
# ==========

def trend_integrity_score(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    high = _series_col(df, "High")
    low = _series_col(df, "Low")
    vol = _series_col(df, "Volume")

    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    # SMA ordering
    cond_sma = (sma20 > sma50) & (sma50 > sma200)
    score_sma = cond_sma.astype(int) * 40

    # Higher highs / higher lows (simple rolling check)
    hh = high > high.shift(1)
    hl = low > low.shift(1)
    trend_hh_hl = (hh & hl).rolling(5).mean()  # fraction of last 5 bars
    score_hh_hl = (trend_hh_hl * 30).clip(0, 30)

    # Volatility contraction (ATR down vs 20-bar mean)
    atr14 = atr(df, 14)
    atr_sma = sma(atr14, 20)
    vol_contract = (atr14 < atr_sma * 0.9).astype(int)  # 10% lower
    score_vol = vol_contract * 20

    # Volume expansion on breakouts (close > 20d high & vol > 20d avg)
    vol_avg = sma(vol, 20)
    breakout = (close > close.rolling(20).max().shift(1)) & (vol > vol_avg * 1.2)
    score_vol_exp = breakout.astype(int) * 10

    return (score_sma + score_hh_hl + score_vol + score_vol_exp).clip(0, 100)


def liquidity_stability_score(df: pd.DataFrame, cfg: EngineConfig) -> pd.Series:
    vol = _series_col(df, "Volume").rolling(20).mean()
    high = _series_col(df, "High")
    low = _series_col(df, "Low")
    close = _series_col(df, "Close")
    open_ = _series_col(df, "Open")

    # Approx spread proxy: (High-Low)/Close
    spread_pct = (high - low) / close * 100
    # Overnight gap: prev close vs today open
    gap_pct = (open_ - close.shift(1)).abs() / close.shift(1) * 100

    good_vol = (vol >= cfg.min_avg_volume)
    good_spread = (spread_pct <= cfg.max_spread_pct)
    good_gap = (gap_pct <= cfg.max_overnight_gap_pct)

    score = (
        good_vol.astype(int) * 40 +
        good_spread.astype(int) * 30 +
        good_gap.astype(int) * 30
    )
    return score.clip(0, 100)


def shock_absorption_score(df: pd.DataFrame) -> pd.Series:
    # Measures how well price recovers after large down bars
    close = _series_col(df, "Close")
    atr14 = atr(df, 14)
    big_down = (close.diff() < -1.0 * atr14)  # large down move
    # Recovery within next 3 bars
    fwd_max = close.shift(-1).rolling(3).max()
    recovered = (fwd_max >= close.shift(1)).fillna(False)
    shock_recovery = (big_down & recovered).rolling(20).mean()  # fraction
    return (shock_recovery * 100).clip(0, 100)


def volatility_compression_score(df: pd.DataFrame) -> pd.Series:
    atr14 = atr(df, 14)
    close = _series_col(df, "Close")
    atr_norm = atr14 / close * 100
    # Lower ATR% => higher score
    max_ref = atr_norm.rolling(50).max()
    score = (1 - (atr_norm / (max_ref + 1e-9))).clip(0, 1) * 100
    return score


def volume_expansion_score(df: pd.DataFrame) -> pd.Series:
    vol = _series_col(df, "Volume")
    vol_avg = sma(vol, 20)
    ratio = (vol / (vol_avg + 1e-9))
    # >1 => expansion
    score = ((ratio - 1).clip(0, 2) / 2) * 100
    return score


def momentum_strength_score(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    rsi5 = rsi(close, 5)
    macd_line, signal_line, hist = macd(close)
    # Normalize RSI and MACD hist
    rsi_score = ((rsi5 - 50).clip(0, 50) / 50) * 60  # 0–60
    hist_norm = hist / (close.rolling(50).std() + 1e-9)
    hist_score = hist_norm.clip(0, 2) / 2 * 40       # 0–40
    return (rsi_score + hist_score).clip(0, 100)


def momentum_quality_score(df: pd.DataFrame, cfg: EngineConfig) -> pd.Series:
    trend_score = trend_integrity_score(df)
    mom_score = momentum_strength_score(df)
    liq_score = liquidity_stability_score(df, cfg)
    vol_comp_score = volatility_compression_score(df)
    vol_exp_score = volume_expansion_score(df)

    mqs = (
        0.25 * trend_score +
        0.25 * mom_score +
        0.20 * liq_score +
        0.15 * vol_comp_score +
        0.15 * vol_exp_score
    )
    return mqs.clip(0, 100)


# ==========
# PHASE CLASSIFICATION
# ==========

def classify_momentum_phase(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    vol = _series_col(df, "Volume")
    rsi5 = rsi(close, 5)
    macd_line, signal_line, hist = macd(close)
    vol_avg = sma(vol, 20)

    phase = pd.Series(index=df.index, dtype="object")

    # Climax: RSI5 > 85 + volume spike
    climax = (rsi5 > 85) & (vol > vol_avg * 1.5)

    # Exhaustion: RSI falling from high + hist weakening
    exhaustion = (rsi5 < 70) & (hist < hist.shift(1)) & (hist > 0)

    # Failure: MACD hist < 0 and close < 21 EMA
    ema21 = ema(close, 21)
    failure = (hist < 0) & (close < ema21)

    # Ignition: hist crosses >0 & close breaks short-term range
    ignition = (hist > 0) & (hist.shift(1) <= 0) & (close > close.rolling(10).max().shift(1))

    # Expansion: hist > 0, RSI > 60, vol > avg
    expansion = (hist > 0) & (rsi5 > 60) & (vol > vol_avg)

    phase[ignition] = "Ignition"
    phase[expansion] = "Expansion"
    phase[climax] = "Climax"
    phase[exhaustion] = "Exhaustion"
    phase[failure] = "Failure"
    phase = phase.fillna("Neutral")
    return phase


# ==========
# ENTRY / EXIT ENGINES
# ==========

def micro_trend_alignment(df: pd.DataFrame, anchor_idx: Optional[int] = None) -> pd.Series:
    close = _series_col(df, "Close")
    low = _series_col(df, "Low")
    ema5 = ema(close, 5)
    ema8 = ema(close, 8)
    ema21 = ema(close, 21)

    if anchor_idx is None:
        # default anchor: last 20-bar low
        rolling_low = low.rolling(20).min()
        if rolling_low.dropna().empty:
            anchor_idx = 0
        else:
            anchor_label = rolling_low.idxmin()
            try:
                anchor_idx = df.index.get_loc(anchor_label)
            except Exception:
                anchor_idx = 0

    vwap = anchored_vwap(df, anchor_idx)

    cond = (ema5 > ema8) & (ema8 > ema21) & (close > vwap)
    return cond


def momentum_pulse_confirmation(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    vol = _series_col(df, "Volume")
    rsi5 = rsi(close, 5)
    macd_line, signal_line, hist = macd(close)
    vol_avg = sma(vol, 20)

    cond_rsi = (rsi5 > 50) & (rsi5 > rsi5.shift(1))
    cond_macd = (hist > 0) & (hist > hist.shift(1))
    cond_vol = vol > vol_avg

    return cond_rsi & cond_macd & cond_vol


def volatility_gate(df: pd.DataFrame, cfg: EngineConfig) -> pd.Series:
    atr14 = atr(df, 14)
    close = _series_col(df, "Close")
    high = _series_col(df, "High")
    low = _series_col(df, "Low")
    atr_pct = atr14 / close * 100
    # True range contraction for 3+ days
    tr = (high - low)
    tr_contract = tr < tr.rolling(10).mean()
    tr_contract_3 = tr_contract.rolling(3).sum() >= 3

    cond_atr = atr_pct < cfg.max_intraday_atr_pct
    return cond_atr & tr_contract_3


def entry_signal(df: pd.DataFrame, cfg: EngineConfig) -> pd.Series:
    micro = micro_trend_alignment(df)
    pulse = momentum_pulse_confirmation(df)
    vol_gate = volatility_gate(df, cfg)
    return micro & pulse & vol_gate


def exit_momentum_fade(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    vol = _series_col(df, "Volume")
    rsi5 = rsi(close, 5)
    macd_line, signal_line, hist = macd(close)
    vol_avg = sma(vol, 20)

    cond_rsi = (rsi5 < 70) & (rsi5 < rsi5.shift(1))
    cond_macd = hist < 0
    cond_vol = vol < vol_avg
    return cond_rsi & cond_macd & cond_vol


def exit_trend_break(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    low = _series_col(df, "Low")
    ema21 = ema(close, 21)
    swing_low = low.rolling(10).min().shift(1)
    cond_ema = close < ema21
    cond_swing = close < swing_low
    return cond_ema | cond_swing


def exit_structural_failure(df: pd.DataFrame) -> pd.Series:
    close = _series_col(df, "Close")
    open_ = _series_col(df, "Open")
    high = _series_col(df, "High")
    low = _series_col(df, "Low")
    atr14 = atr(df, 14)
    body = close - open_
    prev_close = close.shift(1)
    prev_open = open_.shift(1)

    # ATR spike
    atr_spike = atr14 > atr14.rolling(20).mean() * 1.5

    # Bearish engulfing
    prev_body = prev_close - prev_open
    bearish_engulf = (body < 0) & (prev_body > 0) & (open_ > prev_close) & (close < prev_open)

    # VWAP rejection (approx: close < intraday mid + long upper wick)
    upper_wick = high - np.maximum(open_, close)
    vwap_reject = (upper_wick > atr14 * 0.5) & (close < (high + low) / 2)

    return (atr_spike & bearish_engulf) | vwap_reject


def exit_signal(df: pd.DataFrame) -> pd.Series:
    fade = exit_momentum_fade(df)
    trend = exit_trend_break(df)
    structural = exit_structural_failure(df)
    return fade | trend | structural


# ==========
# MULTI-TIMEFRAME + POSITION SIZING
# ==========

def multi_timeframe_alignment(
    daily: pd.DataFrame,
    h4: Optional[pd.DataFrame] = None,
    h1: Optional[pd.DataFrame] = None
) -> pd.Series:
    # Daily trend bullish: 20 > 50 > 200
    d_close = _series_col(daily, "Close")
    d20 = sma(d_close, 20)
    d50 = sma(d_close, 50)
    d200 = sma(d_close, 200)
    daily_bull = (d20 > d50) & (d50 > d200)

    if h4 is not None and not h4.empty:
        h4_close = _series_col(h4, "Close")
        h4_20 = sma(h4_close, 20)
        h4_50 = sma(h4_close, 50)
        h4_bull = h4_20 > h4_50
        # align by last available h4 state
        last_h4_bull = h4_bull.reindex(daily.index, method="ffill")
    else:
        last_h4_bull = pd.Series(True, index=daily.index)

    if h1 is not None and not h1.empty:
        h1_phase = classify_momentum_phase(h1)
        h1_phase_on_daily = h1_phase.reindex(daily.index, method="ffill")
        good_phase = h1_phase_on_daily.isin(["Ignition", "Expansion"])
    else:
        good_phase = pd.Series(True, index=daily.index)

    return daily_bull & last_h4_bull & good_phase


def position_size(
    df: pd.DataFrame,
    cfg: EngineConfig,
    equity: float
) -> pd.Series:
    atr14 = atr(df, 14)
    stop = 2.5 * atr14
    risk_amount = equity * cfg.risk_per_trade
    size = risk_amount / (stop + 1e-9)
    # size in shares
    return size.clip(lower=0)


# ==========
# CATALYST SCORE (EVENT-DRIVEN)
# ==========

def catalyst_score(
    df: pd.DataFrame,
    earnings_dates: Optional[pd.Series] = None,
    analyst_upgrades: Optional[pd.Series] = None,
    sector_rotation_flag: Optional[pd.Series] = None,
    macro_catalyst_flag: Optional[pd.Series] = None,
    short_squeeze_flag: Optional[pd.Series] = None
) -> pd.Series:
    # All optional boolean series aligned to df.index
    idx = df.index
    score = pd.Series(0, index=idx, dtype=float)

    def add_if(series: Optional[pd.Series], pts: float):
        nonlocal score
        if series is not None:
            s = series.reindex(idx).fillna(False).astype(bool)
            score = score + s.astype(int) * pts

    add_if(earnings_dates, 5)
    add_if(analyst_upgrades, 4)
    add_if(sector_rotation_flag, 4)
    add_if(macro_catalyst_flag, 3)
    add_if(short_squeeze_flag, 4)

    return score.clip(0, 20)


# ==========
# NO-TRADE ZONES
# ==========

def no_trade_mask(
    df: pd.DataFrame,
    cfg: EngineConfig,
    earnings_window: Optional[pd.Series] = None,
    vix_series: Optional[pd.Series] = None
) -> pd.Series:
    idx = df.index
    atr14 = atr(df, 14)
    close = _series_col(df, "Close")
    atr_pct = atr14 / close * 100

    too_volatile = atr_pct > cfg.max_atr_pct

    if earnings_window is not None:
        earnings_window = earnings_window.reindex(idx).fillna(False)
    else:
        earnings_window = pd.Series(False, index=idx)

    if vix_series is not None:
        vix_series = vix_series.reindex(idx, method="ffill")
        high_vix = vix_series > cfg.vix_threshold
    else:
        high_vix = pd.Series(False, index=idx)

    # You can add more (gap monsters, low inst. ownership) via extra flags
    return too_volatile | earnings_window | high_vix


# ==========
# AI NARRATIVE ENGINE
# ==========

def build_narrative(
    df: pd.DataFrame,
    mqs: pd.Series,
    phase: pd.Series,
    catalyst: pd.Series
) -> pd.Series:
    close = _series_col(df, "Close")
    trend_score = trend_integrity_score(df)
    mom_score = momentum_strength_score(df)
    liq_score = liquidity_stability_score(df, EngineConfig())
    idx = df.index

    narratives = []
    for i in idx:
        mq = mqs.loc[i]
        ph = phase.loc[i]
        cat = catalyst.loc[i]
        t = trend_score.loc[i]
        ms = mom_score.loc[i]
        ls = liq_score.loc[i]
        px = close.loc[i]

        why = []
        if ms > 60:
            why.append("strong short-term momentum")
        if t > 60:
            why.append("well-structured uptrend")
        if cat >= 8:
            why.append("recent catalysts supporting the move")

        why_str = ", ".join(why) if why else "moderate technical strength"

        support = []
        if t > 70:
            support.append("trend structure is clean with aligned moving averages")
        if ls > 60:
            support.append("liquidity conditions are stable")
        if cat > 0:
            support.append("event-driven flows may be reinforcing price action")

        support_str = " ".join(support) if support else "no major structural tailwinds identified."

        risk = []
        if ph in ["Climax", "Exhaustion"]:
            risk.append("momentum appears late-stage and vulnerable to reversal")
        if mqs.loc[i] < 70:
            risk.append("overall momentum quality is below preferred threshold")
        if cat == 0:
            risk.append("absence of clear catalysts may weaken follow-through")

        risk_str = " ".join(risk) if risk else "no immediate structural red flags."

        next_move = ""
        if ph in ["Ignition", "Expansion"] and mq >= 70:
            next_move = "High probability of continuation if current structure holds."
        elif ph in ["Climax", "Exhaustion"]:
            next_move = "High probability of mean reversion or consolidation."
        elif ph == "Failure":
            next_move = "Trend failure suggests avoiding new longs and considering full exit."
        else:
            next_move = "Wait for clearer phase alignment before committing capital."

        narrative = (
            f"Price at {px:.2f}. Momentum phase: {ph}. "
            f"Momentum quality score: {mq:.1f}. "
            f"Why momentum exists: {why_str}. "
            f"Structural support: {support_str} "
            f"Key vulnerabilities: {risk_str} "
            f"Next likely move: {next_move}"
        )
        narratives.append(narrative)

    return pd.Series(narratives, index=idx)


# ==========
# TOP-LEVEL API
# ==========

def analyze_ticker(
    daily: pd.DataFrame,
    h4: Optional[pd.DataFrame] = None,
    h1: Optional[pd.DataFrame] = None,
    equity: float = 100_000,
    cfg: EngineConfig = EngineConfig(),
    earnings_window: Optional[pd.Series] = None,
    vix_series: Optional[pd.Series] = None,
    earnings_dates: Optional[pd.Series] = None,
    analyst_upgrades: Optional[pd.Series] = None,
    sector_rotation_flag: Optional[pd.Series] = None,
    macro_catalyst_flag: Optional[pd.Series] = None,
    short_squeeze_flag: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    daily / h4 / h1: OHLCV dataframes with columns:
        ['Open','High','Low','Close','Volume']
    All indexed by datetime.
    """

    daily = daily.copy()

    # Core scores
    mqs = momentum_quality_score(daily, cfg)
    phase = classify_momentum_phase(daily)
    entry = entry_signal(daily, cfg)
    exit_ = exit_signal(daily)
    mfa = multi_timeframe_alignment(daily, h4, h1)
    size = position_size(daily, cfg, equity)

    # Catalysts + no-trade
    catalyst = catalyst_score(
        daily,
        earnings_dates=earnings_dates,
        analyst_upgrades=analyst_upgrades,
        sector_rotation_flag=sector_rotation_flag,
        macro_catalyst_flag=macro_catalyst_flag,
        short_squeeze_flag=short_squeeze_flag,
    )
    no_trade = no_trade_mask(daily, cfg, earnings_window, vix_series)

    # Final tradeable long signal
    long_ok = (
        (mqs >= cfg.min_mqs) &
        entry &
        mfa &
        (~no_trade)
    )

    narrative = build_narrative(daily, mqs, phase, catalyst)

    result = {
        "mqs": mqs,
        "phase": phase,
        "entry_signal": entry,
        "exit_signal": exit_,
        "multi_timeframe_ok": mfa,
        "position_size_shares": size.where(long_ok, 0),
        "catalyst_score": catalyst,
        "no_trade_zone": no_trade,
        "long_ok": long_ok,
        "narrative": narrative,
    }
    return result
