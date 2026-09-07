import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.evaluate_judges import factor_alignment


class TestJudgeEvaluation(unittest.TestCase):
    def test_factor_alignment_uses_recorded_direction(self):
        entry = {"factor_evidence": [
            {"name": "RSI", "direction": "🟢"},
            {"name": "量能", "direction": "🔴"},
        ]}
        self.assertEqual(factor_alignment(entry, "SHORT"), ["量能"])

    def test_observation_has_no_alignment(self):
        self.assertEqual(factor_alignment({}, None), [])


if __name__ == "__main__":
    unittest.main()
