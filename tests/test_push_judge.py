import unittest
from unittest.mock import patch

import vt_vote_bot as bot


class TestPushJudge(unittest.TestCase):
    def test_single_judge_pushes_and_multi_not_called(self):
        with patch.object(bot, "PUSH_JUDGE", "single"), patch.object(bot, "SHADOW_JUDGE", False), \
             patch.object(bot, "build_market_brief", return_value="简报"), \
             patch.object(bot, "multi_agent_judge") as multi, \
             patch.object(bot, "_judge", return_value={"direction": "SHORT", "confidence": 62}) as single, \
             patch.object(bot, "audited_decision", side_effect=lambda symbol, result, tf, brief: result):
            j = bot.ai_judge_15m({"symbol": "ETHUSDC"})
        self.assertEqual(j["direction"], "SHORT")
        self.assertIs(single.call_args.args[0], bot.SYS_15M)
        multi.assert_not_called()

    def test_single_4h_uses_tech_prompt_with_prev(self):
        prev = {"direction": "SHORT", "confidence": 62, "mag_tier": 1}
        with patch.object(bot, "PUSH_JUDGE", "single"), patch.object(bot, "SHADOW_JUDGE", False), \
             patch.object(bot, "build_brief_4h", return_value="4h简报"), \
             patch.object(bot, "_judge", return_value={"direction": "SHORT", "confidence": 66}) as single, \
             patch.object(bot, "audited_decision", side_effect=lambda symbol, result, tf, brief: result):
            bot.ai_judge_4h({"symbol": "ETHUSDC"}, prev=prev)
        system, brief = single.call_args.args
        self.assertIs(system, bot.SYS_4H_TECH)
        self.assertIn("上一次4h判决(上根收线): 做空", brief)


class TestEventThrottle(unittest.TestCase):
    def setUp(self):
        bot.EVENT_PUSHED.clear()

    def test_same_kind_event_pushed_once_per_30min(self):
        ev = ["逼近支撑 $2410", "RSI 超卖"]
        self.assertEqual(bot.throttle_key_events("ETH", ev, now=1000), ["逼近支撑 $2410"])
        bot.mark_events_pushed("ETH", ev, now=1000)
        self.assertEqual(bot.throttle_key_events("ETH", ["逼近支撑 $2405"], now=1000 + 600), [])
        # 不同类事件不受影响
        self.assertEqual(bot.throttle_key_events("ETH", ["跌破近20周期低点"], now=1000 + 600),
                         ["跌破近20周期低点"])
        self.assertEqual(bot.throttle_key_events("ETH", ["逼近支撑 $2400"], now=1000 + 1800),
                         ["逼近支撑 $2400"])


if __name__ == "__main__":
    unittest.main()
