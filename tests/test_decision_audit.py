import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import vt_vote_bot as bot
from decision_audit import archive_decision, evaluation_entries, stamp_decision


class TestDecisionAudit(unittest.TestCase):
    def test_new_inference_gets_new_id_without_mutating_previous(self):
        original = {"direction": "LONG", "confidence": 65}
        first = stamp_decision(original, "4h", "当时材料")
        second = stamp_decision(first, "4h", "新的材料")
        self.assertNotIn("decision_meta", original)
        self.assertNotEqual(first["decision_meta"]["id"], second["decision_meta"]["id"])
        self.assertNotEqual(first["decision_meta"]["brief_sha256"], second["decision_meta"]["brief_sha256"])

    def test_archive_is_deduplicated_private_and_preserves_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.sqlite3"
            decision = stamp_decision({"direction": "SHORT", "confidence": 70}, "15m", "原始材料")
            archive_decision(path, "ETHUSDC", decision, "原始材料")
            archive_decision(path, "ETHUSDC", decision, "原始材料")
            self.assertEqual(len(evaluation_entries(path)), 1)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            with sqlite3.connect(path) as database:
                stored = json.loads(database.execute("SELECT payload FROM decisions").fetchone()[0])
            self.assertEqual(stored["brief"], "原始材料")
            self.assertEqual(stored["scope"], "technical_decision_before_macro_veto")

    def test_cache_copies_keep_id_and_macro_changes_do_not_rewrite_archive(self):
        decision = stamp_decision({"direction": "LONG"}, "4h", "brief")
        cached = dict(decision)
        cached["direction"] = "SHORT"
        self.assertEqual(decision["decision_meta"]["id"], cached["decision_meta"]["id"])
        self.assertEqual(decision["direction"], "LONG")

    def test_storage_failure_warns_but_does_not_change_direction(self):
        with mock.patch.object(bot, "archive_decision", side_effect=sqlite3.OperationalError()), mock.patch("builtins.print") as log:
            result = bot.audited_decision("ETHUSDC", {"direction": "LONG"}, "4h", "brief")
        self.assertEqual(result["direction"], "LONG")
        self.assertTrue(result["decision_meta"]["id"])
        self.assertTrue(log.called)


if __name__ == "__main__":
    unittest.main()
