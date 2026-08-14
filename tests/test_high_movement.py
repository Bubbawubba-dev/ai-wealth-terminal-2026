import copy
import unittest
from datetime import datetime, timezone

from high_movement import (
    build_candidate_display_context,
    build_high_movement_payload,
    calculate_risk_reward_ratio,
    classify_high_movement_regime,
    compute_high_movement_score,
    passes_event_gate,
)


def make_candidate(asset="AAA", direction="up", has_trigger=True, recent_returns=None):
    return {
        "asset": asset,
        "volatility_expansion": 80,
        "volume_anomaly": 60,
        "catalyst_strength": 90,
        "order_flow_liquidity": 70,
        "trend_alignment": 65,
        "catalyst_type": "technical",
        "catalyst_summary": "Compression with trigger proximity.",
        "expected_direction": direction,
        "confidence": 72,
        "ideal_entry": {"type": "zone", "min": 100.0, "max": 101.0},
        "stop_loss": 97.5 if direction == "up" else 103.5,
        "profit_target_1": 104.0 if direction == "up" else 97.0,
        "profit_target_2": 106.0 if direction == "up" else 95.0,
        "holding_time": "4-24h",
        "rationale": "Test rationale.",
        "status": "waiting",
        "comparison_note": "Higher movement than core list.",
        "has_upcoming_catalyst": False,
        "has_technical_trigger": has_trigger,
        "liquidity_ok": True,
        "spread_bps": 4.0,
        "_recent_returns": recent_returns if recent_returns is not None else [0.01, 0.02, 0.015, 0.018, 0.017, 0.021],
        "trace": {"source_inputs": ["historical_data"]},
    }


class HighMovementTests(unittest.TestCase):
    def test_score_math_uses_required_weights(self):
        candidate = make_candidate()
        score, breakdown = compute_high_movement_score(candidate)
        self.assertEqual(breakdown["volatility_expansion"], 20.0)
        self.assertEqual(breakdown["volume_anomaly"], 12.0)
        self.assertEqual(breakdown["catalyst_strength"], 22.5)
        self.assertEqual(breakdown["order_flow_liquidity"], 10.5)
        self.assertEqual(breakdown["trend_alignment"], 9.8)
        self.assertEqual(score, 74.8)

    def test_rr_filter_enforces_minimum_rr_floor(self):
        candidate = make_candidate()
        candidate["ideal_entry"] = 100.0
        candidate["stop_loss"] = 99.0
        candidate["profit_target_1"] = 101.0
        candidate["profit_target_2"] = 101.5
        self.assertAlmostEqual(calculate_risk_reward_ratio(candidate), 1.5)
        payload = build_high_movement_payload([candidate], market_shock=35)
        self.assertEqual(len(payload["high_movement_top5"]), 1)
        self.assertGreaterEqual(payload["high_movement_top5"][0]["risk_reward_ratio"], 1.8)

    def test_correlation_cap_filters_third_highly_correlated_same_direction(self):
        corr_cluster = [0.01, 0.02, 0.015, 0.018, 0.017, 0.021, 0.019]
        low_corr = [-0.02, 0.01, -0.015, 0.018, -0.01, 0.004, -0.006]
        candidates = [
            make_candidate("AAA", "up", recent_returns=corr_cluster),
            make_candidate("BBB", "up", recent_returns=corr_cluster),
            make_candidate("CCC", "up", recent_returns=corr_cluster),
            make_candidate("DDD", "down", recent_returns=low_corr),
            make_candidate("EEE", "down", recent_returns=[-0.02, -0.015, -0.01, -0.018, -0.017, -0.012, -0.01]),
            make_candidate("FFF", "up", recent_returns=[0.03, -0.02, 0.01, -0.015, 0.008, -0.01, 0.012]),
        ]
        payload = build_high_movement_payload(candidates, market_shock=35)
        assets = [row["asset"] for row in payload["high_movement_top5"]]
        self.assertEqual(len(assets), 5)
        self.assertNotIn("CCC", assets)
        self.assertNotIn("RESERVE_BACKFILL_USED", " ".join(payload["warnings"]))

    def test_event_gate_logic_requires_event_or_trigger(self):
        blocked = make_candidate("AAA", has_trigger=False)
        self.assertFalse(passes_event_gate(blocked))
        payload = build_high_movement_payload([blocked], market_shock=35)
        self.assertEqual(len(payload["high_movement_top5"]), 1)
        self.assertTrue(any("RESERVE_BACKFILL_USED" in warning for warning in payload["warnings"]))

    def test_regime_adjustment_conditions_confidence_and_target_aggressiveness(self):
        candidate = make_candidate()
        risk_on_payload = build_high_movement_payload([candidate], market_shock=20)
        risk_off_payload = build_high_movement_payload([candidate], market_shock=85)
        risk_on = risk_on_payload["high_movement_top5"][0]
        risk_off = risk_off_payload["high_movement_top5"][0]
        self.assertEqual(classify_high_movement_regime(20)["state"], "risk-on")
        self.assertEqual(classify_high_movement_regime(85)["state"], "risk-off")
        self.assertGreater(risk_on["confidence"], risk_off["confidence"])
        self.assertGreater(risk_on["profit_target_2"], risk_off["profit_target_2"])

    def test_payload_schema_and_guaranteed_five_candidates(self):
        candidates = [
            make_candidate("AAA", recent_returns=[0.01, 0.02, 0.015, 0.018, 0.017, 0.021]),
            make_candidate("BBB", recent_returns=[-0.02, 0.01, -0.015, 0.018, -0.01, 0.004]),
            make_candidate("CCC", recent_returns=[0.03, -0.02, 0.01, -0.015, 0.008, -0.01]),
            make_candidate("DDD", has_trigger=False, recent_returns=[0.005, 0.006, -0.004, 0.003, -0.002, 0.001]),
            make_candidate("EEE", recent_returns=[-0.01, -0.012, 0.008, 0.01, -0.006, 0.004]),
        ]
        payload = build_high_movement_payload(
            candidates,
            market_shock=50,
            generated_at=datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(len(payload["high_movement_top5"]), 5)
        self.assertEqual(payload["generated_at"], "2026-08-14T12:00:00+00:00")
        self.assertIn("regime", payload)
        self.assertIn("filters", payload)
        self.assertIn("warnings", payload)
        self.assertTrue(any("RESERVE_BACKFILL_USED" in warning for warning in payload["warnings"]))
        required_keys = {
            "asset",
            "score",
            "score_breakdown",
            "catalyst_type",
            "catalyst_summary",
            "expected_direction",
            "confidence",
            "ideal_entry",
            "stop_loss",
            "profit_target_1",
            "profit_target_2",
            "risk_reward_ratio",
            "holding_time",
            "rationale",
            "status",
            "comparison_note",
            "trace",
        }
        self.assertTrue(required_keys.issubset(payload["high_movement_top5"][0].keys()))

    def test_display_context_exposes_direction_color_confidence_band_and_status(self):
        ctx = build_candidate_display_context(make_candidate(direction="down"))
        self.assertEqual(ctx["direction_color"], "#dc2626")
        self.assertEqual(ctx["confidence_band"], "medium")
        self.assertEqual(ctx["status_badge"]["label"], "Waiting")

    def test_payload_builder_does_not_mutate_input_candidates(self):
        candidates = [make_candidate("AAA"), make_candidate("BBB", has_trigger=False)]
        snapshot = copy.deepcopy(candidates)
        build_high_movement_payload(candidates, market_shock=35)
        self.assertEqual(candidates, snapshot)


if __name__ == "__main__":
    unittest.main()
