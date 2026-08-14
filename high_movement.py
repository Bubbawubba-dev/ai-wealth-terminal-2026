from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import pandas as pd


HIGH_MOVEMENT_WEIGHTS = {
    "volatility_expansion": 0.25,
    "volume_anomaly": 0.20,
    "catalyst_strength": 0.25,
    "order_flow_liquidity": 0.15,
    "trend_alignment": 0.15,
}

HIGH_MOVEMENT_FILTERS = {
    "min_rr": 1.8,
    "liquidity_filter": True,
    "max_correlated_same_direction": 2,
}

_DIRECTION_COLORS = {"up": "#16a34a", "down": "#dc2626"}
_STATUS_STYLES = {
    "waiting": {"label": "Waiting", "bg": "#334155", "fg": "#e2e8f0"},
    "triggered": {"label": "Triggered", "bg": "#14532d", "fg": "#dcfce7"},
    "invalidated": {"label": "Invalidated", "bg": "#7f1d1d", "fg": "#fee2e2"},
    "completed": {"label": "Completed", "bg": "#1d4ed8", "fg": "#dbeafe"},
}


def _clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    try:
        return max(low, min(float(value), high))
    except (TypeError, ValueError):
        return low


def classify_high_movement_regime(market_shock: float) -> dict[str, Any]:
    if market_shock >= 70:
        return {
            "state": "risk-off",
            "summary": "Risk-off backdrop: tighten upside confidence and favor faster downside follow-through.",
            "confidence_adjustment": {"up": -10, "down": 4},
            "target_multiplier": {"up": 0.92, "down": 1.05},
        }
    if market_shock >= 45:
        return {
            "state": "choppy",
            "summary": "Choppy backdrop: fade target aggressiveness and discount confidence on both sides.",
            "confidence_adjustment": {"up": -6, "down": -6},
            "target_multiplier": {"up": 0.90, "down": 0.90},
        }
    return {
        "state": "risk-on",
        "summary": "Risk-on backdrop: allow more aggressive upside targets and slightly discount shorts.",
        "confidence_adjustment": {"up": 4, "down": -4},
        "target_multiplier": {"up": 1.05, "down": 0.95},
    }


def compute_high_movement_score(candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    breakdown = {}
    for key, weight in HIGH_MOVEMENT_WEIGHTS.items():
        breakdown[key] = round(_clamp(candidate.get(key, 0.0)) * weight, 1)
    return round(sum(breakdown.values()), 1), breakdown


def passes_event_gate(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("has_upcoming_catalyst") or candidate.get("has_technical_trigger"))


def _entry_midpoint(entry: Any) -> float | None:
    if isinstance(entry, dict):
        low = entry.get("min")
        high = entry.get("max")
        if low is None and high is None:
            return None
        if low is None:
            return float(high)
        if high is None:
            return float(low)
        return (float(low) + float(high)) / 2.0
    if entry is None:
        return None
    return float(entry)


def calculate_risk_reward_ratio(candidate: dict[str, Any]) -> float:
    entry = _entry_midpoint(candidate.get("ideal_entry"))
    stop_loss = candidate.get("stop_loss")
    target_2 = candidate.get("profit_target_2")
    target_1 = candidate.get("profit_target_1")
    if entry is None or stop_loss is None or (target_2 is None and target_1 is None):
        return float(candidate.get("risk_reward_ratio", 0.0) or 0.0)
    stop_distance = abs(entry - float(stop_loss))
    reward_distance = abs(float(target_2 if target_2 is not None else target_1) - entry)
    if stop_distance <= 0:
        return 0.0
    return round(reward_distance / stop_distance, 2)


def _scale_target(entry: float | None, target: Any, multiplier: float) -> Any:
    if entry is None or target is None:
        return target
    target = float(target)
    return round(entry + ((target - entry) * multiplier), 2)


def apply_regime_adjustments(candidate: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    adjusted = deepcopy(candidate)
    direction = adjusted.get("expected_direction", "up")
    conf_adj = regime["confidence_adjustment"].get(direction, 0)
    adjusted["confidence"] = int(_clamp(adjusted.get("confidence", 50) + conf_adj, 0, 100))
    entry = _entry_midpoint(adjusted.get("ideal_entry"))
    multiplier = regime["target_multiplier"].get(direction, 1.0)
    adjusted["profit_target_1"] = _scale_target(entry, adjusted.get("profit_target_1"), 1 + ((multiplier - 1) * 0.5))
    adjusted["profit_target_2"] = _scale_target(entry, adjusted.get("profit_target_2"), multiplier)
    adjusted["risk_reward_ratio"] = calculate_risk_reward_ratio(adjusted)
    if entry is not None and adjusted["risk_reward_ratio"] < HIGH_MOVEMENT_FILTERS["min_rr"] and adjusted.get("stop_loss") is not None:
        stop_distance = abs(entry - float(adjusted["stop_loss"]))
        min_reward_distance = stop_distance * HIGH_MOVEMENT_FILTERS["min_rr"]
        if direction == "up":
            adjusted["profit_target_2"] = round(entry + min_reward_distance, 2)
        else:
            adjusted["profit_target_2"] = round(entry - min_reward_distance, 2)
        adjusted["risk_reward_ratio"] = calculate_risk_reward_ratio(adjusted)
    return adjusted


def direction_color(direction: str) -> str:
    return _DIRECTION_COLORS.get(direction, "#94a3b8")


def confidence_band(confidence: float) -> str:
    confidence = _clamp(confidence, 0, 100)
    if confidence >= 75:
        return "high"
    if confidence >= 55:
        return "medium"
    return "low"


def status_badge(status: str) -> dict[str, str]:
    return _STATUS_STYLES.get(status, _STATUS_STYLES["waiting"])


def build_candidate_display_context(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "direction_color": direction_color(candidate.get("expected_direction", "")),
        "confidence_band": confidence_band(candidate.get("confidence", 0)),
        "status_badge": status_badge(candidate.get("status", "waiting")),
    }


def _pairwise_correlation(left: list[float] | tuple[float, ...] | None, right: list[float] | tuple[float, ...] | None) -> float:
    if not left or not right:
        return 0.0
    left_s = pd.Series(left).dropna()
    right_s = pd.Series(right).dropna()
    lookback = min(len(left_s), len(right_s), 20)
    if lookback < 5:
        return 0.0
    corr = float(left_s.tail(lookback).corr(right_s.tail(lookback)))
    if pd.isna(corr):
        return 0.0
    return corr


def _validate_candidate(candidate: dict[str, Any]) -> list[str]:
    reasons = []
    if candidate.get("expected_direction") not in {"up", "down"}:
        reasons.append("INVALID_DIRECTION")
    if not passes_event_gate(candidate):
        reasons.append("EVENT_GATE_FAILED")
    if candidate.get("stop_loss") in {None, ""}:
        reasons.append("MISSING_STOP_LOSS")
    if HIGH_MOVEMENT_FILTERS["liquidity_filter"]:
        if not candidate.get("liquidity_ok", False):
            reasons.append("LOW_LIQUIDITY")
        if float(candidate.get("spread_bps", 999.0) or 999.0) > 12.0:
            reasons.append("HIGH_SPREAD")
    rr = calculate_risk_reward_ratio(candidate)
    if rr < HIGH_MOVEMENT_FILTERS["min_rr"]:
        reasons.append("RR_BELOW_MIN")
    return reasons


def _hard_rejections(reasons: list[str]) -> list[str]:
    hard_reason_codes = {"INVALID_DIRECTION", "MISSING_STOP_LOSS", "LOW_LIQUIDITY", "HIGH_SPREAD", "RR_BELOW_MIN", "CORRELATION_CAP"}
    return [reason for reason in reasons if reason in hard_reason_codes]


def build_high_movement_payload(
    candidates: list[dict[str, Any]],
    market_shock: float,
    generated_at: datetime | None = None,
    source_inputs: list[str] | None = None,
) -> dict[str, Any]:
    regime = classify_high_movement_regime(market_shock)
    generated_at = generated_at or datetime.now(timezone.utc)
    prepared = []
    rejection_counts: dict[str, int] = {}

    for raw_candidate in deepcopy(candidates):
        candidate = apply_regime_adjustments(raw_candidate, regime)
        score, score_breakdown = compute_high_movement_score(candidate)
        candidate["score"] = score
        candidate["score_breakdown"] = score_breakdown
        candidate["risk_reward_ratio"] = calculate_risk_reward_ratio(candidate)
        prepared.append(candidate)

    prepared.sort(key=lambda row: (-float(row.get("score", 0.0)), row.get("asset", "")))
    selected = []
    selected_assets = set()
    reserve_pool = []
    for candidate in prepared:
        asset = candidate.get("asset")
        if asset in selected_assets:
            continue
        reasons = _validate_candidate(candidate)
        correlation_hit = False
        if not _hard_rejections(reasons):
            same_direction_correlated = 0
            for existing in selected:
                if existing.get("expected_direction") != candidate.get("expected_direction"):
                    continue
                corr = _pairwise_correlation(existing.get("_recent_returns"), candidate.get("_recent_returns"))
                if corr >= 0.80:
                    same_direction_correlated += 1
            correlation_hit = same_direction_correlated >= HIGH_MOVEMENT_FILTERS["max_correlated_same_direction"]
            if correlation_hit:
                reasons.append("CORRELATION_CAP")
        hard_reasons = _hard_rejections(reasons)

        if hard_reasons:
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        if reasons:
            reserve_candidate = deepcopy(candidate)
            reserve_candidate["status"] = "waiting"
            reserve_candidate["confidence"] = int(_clamp(reserve_candidate.get("confidence", 50) - 8, 0, 100))
            reserve_note = "Reserve backfill used to maintain five high-movement names while keeping hard risk filters intact."
            reserve_candidate["comparison_note"] = (
                f"{reserve_candidate.get('comparison_note', '').strip()} {reserve_note}"
            ).strip()
            reserve_candidate["reserve_reason_codes"] = reasons.copy()
            reserve_pool.append(reserve_candidate)
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        selected.append(candidate)
        selected_assets.add(asset)
        if len(selected) >= 5:
            break

    if len(selected) < 5:
        reserve_pool.sort(key=lambda row: (len(row.get("reserve_reason_codes", [])), -float(row.get("score", 0.0)), row.get("asset", "")))
        for reserve_candidate in reserve_pool:
            asset = reserve_candidate.get("asset")
            if asset in selected_assets:
                continue
            selected.append(reserve_candidate)
            selected_assets.add(asset)
            if len(selected) >= 5:
                break

    fallback_used = False
    if len(selected) < 5:
        for candidate in prepared:
            if len(selected) >= 5:
                break
            asset = candidate.get("asset")
            if asset in selected_assets:
                continue
            reasons = _validate_candidate(candidate)
            if _hard_rejections(reasons):
                continue
            if reasons:
                fallback_candidate = deepcopy(candidate)
                fallback_candidate["status"] = "waiting"
                fallback_candidate["confidence"] = int(_clamp(fallback_candidate.get("confidence", 50) - 8, 0, 100))
                fallback_note = "Final soft-filter fallback used after reserve pass to maintain five high-movement names."
                fallback_candidate["comparison_note"] = (
                    f"{fallback_candidate.get('comparison_note', '').strip()} {fallback_note}"
                ).strip()
                fallback_candidate["reserve_reason_codes"] = reasons.copy()
                selected.append(fallback_candidate)
            else:
                selected.append(candidate)
            selected_assets.add(asset)
            fallback_used = True

    warnings = []
    reserve_used = any(candidate.get("reserve_reason_codes") for candidate in selected)
    if reserve_used:
        warnings.append("RESERVE_BACKFILL_USED: one or more names missed a soft filter but passed all hard risk filters.")
    if fallback_used:
        warnings.append("SOFT_FILTER_FALLBACK_USED: deterministic fallback promoted additional soft-filter candidates to keep five names.")
    if len(selected) < 5:
        reason_codes = ", ".join(sorted(rejection_counts)) if rejection_counts else "INSUFFICIENT_QUALIFIED_CANDIDATES"
        warnings.append(f"LESS_THAN_5_VALID_CANDIDATES: {reason_codes}")

    high_movement_top5 = []
    for candidate in selected:
        view = {
            "asset": candidate.get("asset"),
            "score": candidate.get("score"),
            "score_breakdown": candidate.get("score_breakdown"),
            "catalyst_type": candidate.get("catalyst_type", "technical"),
            "catalyst_summary": candidate.get("catalyst_summary", ""),
            "expected_direction": candidate.get("expected_direction"),
            "confidence": candidate.get("confidence"),
            "ideal_entry": candidate.get("ideal_entry"),
            "stop_loss": candidate.get("stop_loss"),
            "profit_target_1": candidate.get("profit_target_1"),
            "profit_target_2": candidate.get("profit_target_2"),
            "risk_reward_ratio": candidate.get("risk_reward_ratio"),
            "holding_time": candidate.get("holding_time", "4-24h"),
            "rationale": candidate.get("rationale", ""),
            "status": candidate.get("status", "waiting"),
            "comparison_note": candidate.get("comparison_note", ""),
            "trace": candidate.get("trace", {}),
        }
        high_movement_top5.append(view)

    return {
        "generated_at": generated_at.isoformat(),
        "regime": {"state": regime["state"], "summary": regime["summary"]},
        "high_movement_top5": high_movement_top5,
        "filters": HIGH_MOVEMENT_FILTERS.copy(),
        "warnings": warnings,
        "trace_metadata": {
            "weights": HIGH_MOVEMENT_WEIGHTS.copy(),
            "feature_sources": source_inputs or ["historical_data", "intraday_5m", "fundamental_cache"],
            "rejection_counts": rejection_counts,
        },
    }
