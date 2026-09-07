import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vt_vote_bot as V


class TestDecisionPipeline(unittest.TestCase):
    def test_first_opposite_close_holds_previous_direction(self):
        previous = {"direction": "LONG", "confidence": 72, "summary": "维持多", "reasons": []}
        candidate = {"direction": "SHORT", "confidence": 78, "summary": "转空", "reasons": ["跌破"]}

        result = V.stabilize_judge(candidate, previous)

        self.assertEqual(result["direction"], "LONG")
        self.assertEqual(result["pending_direction"], "SHORT")
        self.assertIn("等待下一根收线", result["summary"])

    def test_second_opposite_close_confirms_reversal(self):
        previous = {"direction": "LONG", "confidence": 62, "pending_direction": "SHORT"}
        candidate = {"direction": "SHORT", "confidence": 76, "summary": "确认转空", "reasons": ["延续"]}

        result = V.stabilize_judge(candidate, previous)

        self.assertEqual(result["direction"], "SHORT")
        self.assertNotIn("pending_direction", result)

    def test_research_report_is_attached_to_final_decision(self):
        research = {"bull": {"strength": 35}, "bear": {"strength": 70}, "risk": "假跌破"}
        decision = {"direction": "SHORT", "confidence": 71, "reasons": ["结构向下"]}
        with mock.patch.object(V, "_research_call", return_value=research), \
             mock.patch.object(V, "_judge", return_value=decision) as judge:
            result = V.multi_agent_judge("brief")

        self.assertEqual(result["debate"], research)
        self.assertEqual(judge.call_args.args[0], V.MANAGER_SYSTEM)

    def test_research_failure_uses_manager_without_old_judge(self):
        fallback = {"direction": None, "confidence": -1, "reasons": ["不可用"]}
        with mock.patch.object(V, "_research_call", return_value=None), \
             mock.patch.object(V, "_judge", return_value=fallback) as judge:
            result = V.multi_agent_judge("brief")

        self.assertIs(result, fallback)
        self.assertEqual(judge.call_args.args[0], V.MANAGER_SYSTEM)
        self.assertIn("研究组不可用", judge.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
