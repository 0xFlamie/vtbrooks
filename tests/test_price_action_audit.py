import unittest

from analysis.audit_price_action import classify, readiness
from tests.test_brooks_evidence import bars


class TestPriceActionAudit(unittest.TestCase):
    def test_disagreement_does_not_become_an_expert_label(self):
        before = {"legacy": {"state": "trend_up"}, "structure": {"direction": 0}}
        after = {"quality": {"blocked": False}, "structure": {"direction": 0}}
        result = classify(before, after, True)
        self.assertEqual(result["category"], "definition_difference")
        self.assertIsNone(result["human_label"])

    def test_unreliable_inputs_are_separated_from_semantic_disagreement(self):
        before = {"legacy": {"state": "trend_up"}, "structure": {"direction": -1}}
        after = {"quality": {"blocked": True}, "structure": {"direction": -1}}
        self.assertEqual(classify(before, after, False)["category"], "current_input_unreliable")

    def test_training_filter_is_causal_and_does_not_compress_time(self):
        frame = bars()
        before, _ = readiness(frame)
        frame.loc[frame.index[-1], "volume"] = 0
        after, _ = readiness(frame)
        self.assertEqual(len(after), len(frame))
        self.assertEqual(before.iloc[:-1].tolist(), after.iloc[:-1].tolist())
        self.assertFalse(after.iloc[-1])


if __name__ == "__main__":
    unittest.main()
