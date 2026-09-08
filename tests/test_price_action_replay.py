import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.build_price_action_replay import build_payload, load_pair, render_html
from price_action_replay import confirmed_swings, followup, inspect_bar, replay_clip


def bars(count=200, frequency="4h"):
    rng = np.random.default_rng(12)
    close = 100 + np.cumsum(rng.normal(0, .2, count))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": opening, "high": np.maximum(close, opening) + .1,
                         "low": np.minimum(close, opening) - .1, "close": close, "volume": 10.},
                        index=pd.date_range("2024-01-01", periods=count, freq=frequency, tz="UTC"))


class TestReplay(unittest.TestCase):
    def test_inspection_cannot_see_future(self):
        for timeframe in ("4h", "15m"):
            frame = bars(frequency="15min" if timeframe == "15m" else "4h")
            before = inspect_bar(frame, 140, timeframe)
            changed = frame.copy()
            changed.iloc[141:, :4] *= 2
            self.assertEqual(before, inspect_bar(changed, 140, timeframe))
            self.assertNotEqual(followup(frame, 140, before), followup(changed, 140, before))

    def test_swings_need_two_right_bars(self):
        frame = bars(120)
        frame.iloc[118, frame.columns.get_loc("high")] = 1000
        swings = confirmed_swings(frame)
        self.assertFalse(any(s["time"] == frame.index[118].isoformat() for s in swings))
        self.assertTrue(all(pd.Timestamp(s["confirmed_at"]) <= frame.index[-1] for s in swings))

    def test_clip_cursors_and_separate_future_answers(self):
        result = replay_clip(bars(), 130, 10, "4h")
        self.assertEqual(result["offset"], 119)
        self.assertEqual(len(result["snapshots"]), 10)
        self.assertEqual(len(result["bars"]), 132)
        for i, snapshot in enumerate(result["snapshots"]):
            self.assertEqual(snapshot["time"], result["bars"][119+i]["time"])
            self.assertNotIn("outcomes", snapshot)
        self.assertEqual(result["answers"][0]["horizon"], 3)

    def test_missing_future_returns_unavailable(self):
        frame = bars()
        snap = inspect_bar(frame, 198, "15m")
        self.assertEqual(followup(frame, 198, snap), {"available": False})

    def test_flat_candle_ratios_are_missing_not_zero_or_nan(self):
        frame = bars()
        price = frame.close.iloc[140]
        frame.iloc[140, :4] = price
        snap = inspect_bar(frame, 140, "4h")
        self.assertIsNone(snap["body_ratio"])
        self.assertIsNone(snap["overlap"])
        self.assertEqual(snap["flat_bars_20"], 1)
        json.dumps(snap, allow_nan=False)

    def test_incomplete_clip_is_rejected(self):
        with self.assertRaises(ValueError):
            replay_clip(bars(), 195, 10, "4h")


class TestArtifact(unittest.TestCase):
    def test_resampling_uses_only_complete_four_hour_bars(self):
        frame = bars(16 * 140 + 5, "15min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            frame.to_csv(path, index_label="ts")
            pair = load_pair(path)
            self.assertEqual(len(pair["4h"]), 140)
            self.assertAlmostEqual(pair["4h"].iloc[0].close, frame.iloc[15].close)
            self.assertEqual(pair["4h"].iloc[0].volume, frame.iloc[:16].volume.sum())
            payload = build_payload(path, "fixture", "TEST-PAIR", clips=2, steps=2)
            self.assertEqual(len(payload["timeframes"]["4h"]["clips"]), 2)
            self.assertEqual(len(payload["source_sha256"]), 64)
            self.assertEqual(payload["timeframes"]["15m"]["flat_bars"], 0)

    def test_invalid_data_gaps_are_rejected(self):
        frame = bars(32, "15min").drop(bars(32, "15min").index[10])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bars.csv"
            frame.to_csv(path, index_label="ts")
            with self.assertRaisesRegex(ValueError, "缺口"):
                load_pair(path)

    def test_embedded_json_cannot_close_script(self):
        payload = {"source": "</script><script>alert(1)</script>"}
        rendered = render_html(payload)
        self.assertNotIn(payload["source"], rendered)
        embedded = rendered.split('id="replay-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(embedded), payload)


if __name__ == "__main__":
    unittest.main()
