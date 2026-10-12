import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import signal_tracker as st


class TrackerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = patch.object(st, "DB_FILE", os.path.join(self.tmp.name, "scores.sqlite3"))
        self.db.start()
        self.sent = []
        st.set_sender(self.sent.append)
        st._health.update(fails=0, alerted=False, last_reason="")

    def tearDown(self):
        self.db.stop()
        self.tmp.cleanup()


class TestScoring(TrackerCase):
    def test_failed_judgment_is_not_recorded(self):
        st.record("主裁判", "15m", "ETHUSDC", {"direction": None, "confidence": -1}, ts=1000)
        self.assertIn("没有判决记录", st.build_report(7, now=2000))

    def test_settle_and_report_against_baseline(self):
        t0 = 1_700_000_040  # 整分钟
        prices = {t0: 100.0, t0 + 3600: 101.0, t0 + 14400: 99.0,
                  t0 + 60: 100.0, t0 + 60 + 3600: 99.5, t0 + 60 + 14400: 98.0}
        st.record("主裁判", "15m", "ETHUSDC", {"direction": "LONG", "confidence": 65}, ts=t0)
        st.record("主裁判", "15m", "ETHUSDC", {"direction": "SHORT", "confidence": 40}, ts=t0 + 60)
        st.settle(price_at=lambda ts: prices.get(int(ts // 60 * 60)), now=t0 + 5 * 3600)
        text = st.build_report(7, now=t0 + 5 * 3600)
        # 1h: 做多+1% 命中, 做空-0.5% 命中; 恒多只对一半
        self.assertIn("1h后: 命中100%(2/2)", text)
        self.assertIn("恒多50%", text)
        # 4h: 做多-1% 错, 做空-2% 对
        self.assertIn("4h后: 命中50%(1/2)", text)
        self.assertIn("置信≥60 命中100%(1次)", text)

    def test_unexpired_rows_wait(self):
        t0 = 1_700_000_000
        st.record("主裁判", "4h", "ETHUSDC", {"direction": "LONG", "confidence": 70}, ts=t0)
        st.settle(price_at=lambda ts: 100.0, now=t0 + 600)
        self.assertIn("尚未到期", st.build_report(7, now=t0 + 600))

    def test_weekly_report_once_per_week(self):
        monday = datetime(2026, 10, 19, 9, 5, tzinfo=st.BJ).timestamp()
        st.record("主裁判", "15m", "ETHUSDC", {"direction": "LONG", "confidence": 60}, ts=monday - 86400)
        self.assertTrue(st.maybe_weekly_report(now=monday))
        self.assertFalse(st.maybe_weekly_report(now=monday + 600))
        self.assertFalse(st.maybe_weekly_report(now=monday + 86400))  # 周二不发
        self.assertEqual(len(self.sent), 1)


class TestHealth(TrackerCase):
    def test_alert_once_after_consecutive_failures_then_recover(self):
        for _ in range(5):
            st.judge_fail("DS", "HTTP 402 Insufficient Balance")
        self.assertEqual(len(self.sent), 1)
        self.assertIn("连续失败 3 次", self.sent[0])
        st.judge_ok("DS", "deepseek-flash")
        self.assertIn("已恢复", self.sent[1])
        st.judge_fail("DS", "超时")
        self.assertEqual(len(self.sent), 2)

    def test_model_change_alert(self):
        st.judge_ok("DS", "deepseek-chat")
        st.judge_ok("DS", "deepseek-chat")
        self.assertEqual(self.sent, [])
        st.judge_ok("DS", "deepseek-flash")
        self.assertEqual(len(self.sent), 1)
        self.assertIn("deepseek-chat → deepseek-flash", self.sent[0])

    def test_low_balance_alert_throttled(self):
        class R:
            def json(self):
                return {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "3.20"}]}
        with patch.object(st.requests, "get", return_value=R()):
            self.assertEqual(st.check_balance("k", now=10_000), 3.2)
            st.check_balance("k", now=10_000 + 3700)      # 1小时后再查, 12小时内不重复告警
            st.check_balance("k", now=10_000 + 13 * 3600)
        self.assertEqual(len(self.sent), 2)
        self.assertIn("¥3.20", self.sent[0])


if __name__ == "__main__":
    unittest.main()
