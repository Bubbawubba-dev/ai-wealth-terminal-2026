import numpy as np
import pandas as pd


def compute_ripster_cloud(close, fast_spans=(8, 21), slow_spans=(34, 55)):
    close = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fast_a = close.ewm(span=fast_spans[0], adjust=False).mean()
    fast_b = close.ewm(span=fast_spans[1], adjust=False).mean()
    slow_a = close.ewm(span=slow_spans[0], adjust=False).mean()
    slow_b = close.ewm(span=slow_spans[1], adjust=False).mean()
    fast_upper = pd.concat([fast_a, fast_b], axis=1).max(axis=1)
    fast_lower = pd.concat([fast_a, fast_b], axis=1).min(axis=1)
    slow_upper = pd.concat([slow_a, slow_b], axis=1).max(axis=1)
    slow_lower = pd.concat([slow_a, slow_b], axis=1).min(axis=1)
    return {
        "fast_upper": fast_upper,
        "fast_lower": fast_lower,
        "slow_upper": slow_upper,
        "slow_lower": slow_lower,
        "fast_mid": (fast_upper + fast_lower) / 2,
        "slow_mid": (slow_upper + slow_lower) / 2,
    }


def _recent_slope(values):
    recent = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().tail(6)
    return float(recent.diff().dropna().mean()) if len(recent) >= 2 else 0.0


def classify_cloud_state(price, cloud):
    if price is None or len(price) == 0:
        return "insufficient data"

    latest = {}
    for key in ("fast_upper", "fast_lower", "slow_upper", "slow_lower", "fast_mid", "slow_mid"):
        values = pd.to_numeric(cloud[key], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.empty or not np.isfinite(values.iloc[-1]):
            return "insufficient data"
        latest[key] = float(values.iloc[-1])

    prices = pd.to_numeric(price, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if prices.empty or not np.isfinite(prices.iloc[-1]):
        return "insufficient data"
    px = float(prices.iloc[-1])
    if px <= 0:
        return "insufficient data"

    fast_slope = _recent_slope(cloud["fast_mid"])
    slow_slope = _recent_slope(cloud["slow_mid"])
    fast_width = max(latest["fast_upper"] - latest["fast_lower"], 0.0)
    slow_width = max(latest["slow_upper"] - latest["slow_lower"], 0.0)
    compressed = (fast_width + slow_width) / px * 100 < 0.6

    if (
        px > latest["fast_upper"]
        and latest["fast_mid"] > latest["slow_mid"]
        and fast_slope > 0
        and slow_slope > 0
    ):
        return "bullish trend"
    if (
        px < latest["fast_lower"]
        and latest["fast_mid"] < latest["slow_mid"]
        and fast_slope < 0
        and slow_slope < 0
    ):
        return "bearish trend"
    if compressed:
        return "compression/chop"

    bullish_cloud = latest["fast_mid"] > latest["slow_mid"] and fast_slope > 0 and slow_slope > 0
    bearish_cloud = latest["fast_mid"] < latest["slow_mid"] and fast_slope < 0 and slow_slope < 0
    inside_cloud = (
        latest["fast_lower"] <= px <= latest["fast_upper"]
        or latest["slow_lower"] <= px <= latest["slow_upper"]
    )
    if bullish_cloud or bearish_cloud or inside_cloud or np.sign(fast_slope) != np.sign(slow_slope):
        return "transition"
    return "compression/chop"


def cloud_quality_metrics(close, cloud):
    prices = pd.to_numeric(close, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if prices.empty or not np.isfinite(prices.iloc[-1]):
        return {"separation_pct": 0.0, "slope_strength_pct": 0.0, "expansion_ratio": 0.0, "flip_count": 0}
    if any(
        values.empty or not np.isfinite(values.iloc[-1])
        for values in (cloud["fast_mid"], cloud["slow_mid"], cloud["fast_upper"], cloud["fast_lower"], cloud["slow_upper"], cloud["slow_lower"])
    ):
        return {"separation_pct": 0.0, "slope_strength_pct": 0.0, "expansion_ratio": 0.0, "flip_count": 0}
    px = max(abs(float(prices.iloc[-1])), 1e-9)
    fast_mid = cloud["fast_mid"]
    slow_mid = cloud["slow_mid"]
    fast_width = cloud["fast_upper"] - cloud["fast_lower"]
    slow_width = cloud["slow_upper"] - cloud["slow_lower"]
    cloud_sep_pct = abs(float(fast_mid.iloc[-1] - slow_mid.iloc[-1])) / px * 100
    slope_strength = (abs(_recent_slope(fast_mid)) + abs(_recent_slope(slow_mid))) / px * 100
    total_width = (fast_width + slow_width).replace([np.inf, -np.inf], np.nan)
    expansion_now = max(float(total_width.iloc[-1]), 0.0) if len(total_width) else 0.0
    previous_width = total_width.iloc[-21:-1].dropna()
    expansion_prev = float(previous_width.mean()) if not previous_width.empty else expansion_now
    expansion_ratio = expansion_now / (expansion_prev + 1e-9)
    directions = np.sign(pd.to_numeric(fast_mid, errors="coerce").diff().tail(20)).dropna()
    directions = directions[directions != 0]
    flip_count = int((directions != directions.shift()).sum() - 1) if len(directions) > 1 else 0
    return {
        "separation_pct": cloud_sep_pct,
        "slope_strength_pct": slope_strength,
        "expansion_ratio": expansion_ratio,
        "flip_count": max(flip_count, 0),
    }


def cloud_confidence_score(cloud_state, cloud_metrics):
    sep_score = float(np.interp(cloud_metrics["separation_pct"], [0.05, 0.2, 0.8], [15, 55, 95]))
    slope_score = float(np.interp(cloud_metrics["slope_strength_pct"], [0.01, 0.05, 0.2], [10, 50, 95]))
    expansion_score = float(np.interp(cloud_metrics["expansion_ratio"], [0.7, 1.0, 1.4], [25, 60, 95]))
    raw = sep_score * 0.4 + slope_score * 0.35 + expansion_score * 0.25
    if cloud_state in {"compression/chop", "insufficient data"}:
        raw -= 20
    if cloud_state == "transition":
        raw -= 14
    if cloud_metrics["flip_count"] >= 4:
        raw -= 10
    return float(np.clip(raw, 0, 100))
