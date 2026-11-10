import unittest

import numpy as np
import pandas as pd

from ripster_cloud import (
    classify_cloud_state,
    cloud_confidence_score,
    cloud_quality_metrics,
    compute_ripster_cloud,
)


class RipsterCloudTests(unittest.TestCase):
    def test_cloud_classifies_rising_and_falling_trends(self):
        rising = pd.Series(np.linspace(100, 150, 100))
        falling = pd.Series(np.linspace(150, 100, 100))

        self.assertEqual(classify_cloud_state(rising, compute_ripster_cloud(rising)), "bullish trend")
        self.assertEqual(classify_cloud_state(falling, compute_ripster_cloud(falling)), "bearish trend")

    def test_price_inside_directional_cloud_is_a_transition(self):
        rising = pd.Series(np.linspace(100, 150, 100))
        cloud = compute_ripster_cloud(rising)
        price = rising.copy()
        price.iloc[-1] = (cloud["fast_upper"].iloc[-1] + cloud["fast_lower"].iloc[-1]) / 2

        self.assertEqual(classify_cloud_state(price, cloud), "transition")

    def test_flat_series_is_compression_and_metrics_are_finite(self):
        close = pd.Series([100.0] * 60)
        cloud = compute_ripster_cloud(close)
        metrics = cloud_quality_metrics(close, cloud)

        self.assertEqual(classify_cloud_state(close, cloud), "compression/chop")
        self.assertEqual(metrics["flip_count"], 0)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreaterEqual(cloud_confidence_score("compression/chop", metrics), 0)

    def test_missing_latest_price_is_reported_as_insufficient_data(self):
        close = pd.Series([100.0, 101.0, np.nan])

        self.assertEqual(classify_cloud_state(close, compute_ripster_cloud(close)), "insufficient data")
        self.assertEqual(
            cloud_quality_metrics(pd.Series(dtype=float), compute_ripster_cloud(pd.Series(dtype=float))),
            {"separation_pct": 0.0, "slope_strength_pct": 0.0, "expansion_ratio": 0.0, "flip_count": 0},
        )


if __name__ == "__main__":
    unittest.main()
