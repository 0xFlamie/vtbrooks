from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from web.research_signals import RULE_SHA256, STRATEGY_ID, read_snapshot

ROOT = Path(__file__).resolve().parents[1]


class TestResearchSignals(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "shadow.sqlite3"
        self.at = datetime(2026, 9, 10, 9, 0, tzinfo=timezone.utc)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE events(seq INTEGER PRIMARY KEY,event_key TEXT UNIQUE,payload TEXT,sha256 TEXT)")
        self.add("protocol", "protocol", {"policy": {"version": "failed-range-prospective-shadow-v2",
                 "venue": "Coinbase", "symbol": "ETH-USD"},
                 "implementation_sha256": {"brooks_failed_range.py": RULE_SHA256}})
        self.add("heartbeat:test:now", "heartbeat", {"received_at": self.at.isoformat()})

    def add(self, key, kind, payload, recorded=None):
        with sqlite3.connect(self.path) as db:
            tail = db.execute("SELECT sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            value = {"key": key, "kind": kind, "payload": payload,
                     "recorded_at": (recorded or self.at).isoformat(), "previous_sha256": tail[0] if tail else None}
            encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT INTO events(event_key,payload,sha256) VALUES (?,?,?)",
                       (key, encoded, hashlib.sha256(encoded.encode()).hexdigest()))

    def signal(self, side=1, eligible=True, generated=None):
        generated = generated or self.at
        available = generated.replace(second=0, microsecond=0)
        episode = "failed_range:" + (available - timedelta(minutes=3)).isoformat()
        snapshot = {"prospective_eligible": eligible, "generated_at": generated.isoformat(),
                    "entry_not_before": (available + timedelta(minutes=2)).isoformat(),
                    "deadline": (available + timedelta(hours=1)).isoformat(),
                    "signal": {"episode_id": episode, "available_at": available.isoformat(),
                               "family": "failed_range", "direction": side, "observed_price": 100.,
                               "atr": 2., "range_low": 95., "range_high": 105., "structure_reference": 94.}}
        self.add("signal:" + episode, "signal", snapshot, generated)
        return episode, snapshot

    def view(self, minutes=0):
        return read_snapshot(self.path, self.at + timedelta(minutes=minutes))

    def test_empty_healthy_ledger_not_fabricated_as_signal(self):
        value = self.view()
        self.assertEqual(value["collector"]["status"], "live")
        self.assertEqual(value["signals"], [])
        self.assertEqual(value["strategy_id"], STRATEGY_ID)
        self.assertEqual(value["source"], "Coinbase ETH-USD")

    def test_projection_does_not_write_or_expose_private_payload(self):
        episode, _ = self.signal()
        before = self.path.read_bytes()
        value = self.view()
        self.assertEqual(self.path.read_bytes(), before)
        signal = value["signals"][0]
        self.assertEqual(signal["id"], STRATEGY_ID + ":" + episode)
        self.assertEqual(signal["status"], "new")
        self.assertNotIn("implementation_sha256", json.dumps(value))
        self.assertNotIn("trades", signal)
        self.assertNotIn("entry_price", signal)

    def test_short_direction_and_explanation(self):
        self.signal(-1)
        signal = self.view()["signals"][0]
        self.assertEqual(signal["direction"], "SHORT")
        self.assertIn("向下", signal["reason"])

    def test_boolean_direction_is_not_a_long_signal(self):
        self.signal(side=True)
        self.assertEqual(self.view()["signals"], [])
        self.assertEqual(self.view()["collector"]["status"], "unavailable")

    def test_observing_and_ended_not_new_entry_alerts(self):
        self.signal()
        self.assertEqual(self.view(2)["signals"][0]["status"], "observing")
        self.assertEqual(self.view(60)["signals"][0]["status"], "ended")

    def test_expired_heartbeat_marks_active_candidate_unverified(self):
        self.signal()
        value = self.view(4)
        self.assertEqual(value["collector"]["status"], "stale")
        self.assertEqual(value["signals"][0]["status"], "unverified")

    def test_failure_overrides_recent_heartbeat(self):
        self.signal()
        self.add("error:one", "capture_failed", {"error_type": "ValueError"})
        self.assertEqual(self.view()["collector"]["status"], "stale")

    def test_reconnect_is_not_claimed_healthy_before_new_heartbeat(self):
        self.add("start:next", "capture_start", {})
        self.assertEqual(self.view()["collector"]["status"], "warming")

    def test_ineligible_candidates_are_not_signals(self):
        self.signal(eligible=False)
        self.assertEqual(self.view()["signals"], [])

    def test_unknown_settlement_not_a_loss_or_win(self):
        episode, _ = self.signal()
        self.add("settle:signal:" + episode, "settlement", {"status": "unverified"})
        self.assertEqual(self.view(65)["signals"][0]["result"], "后续数据不足，未判定")

    def test_held_space_not_labelled_profit(self):
        episode, _ = self.signal()
        self.add("settle:signal:" + episode, "settlement",
                 {"status": "evaluated", "cases": {"0.25/3": {"forward": {"held": 1}}}})
        self.assertIn("非盈利判定", self.view(65)["signals"][0]["result"])

    def test_missing_database_not_created(self):
        missing = Path(self.temp.name) / "missing.sqlite3"
        self.assertEqual(read_snapshot(missing, self.at)["collector"]["status"], "unavailable")
        self.assertFalse(missing.exists())

    def test_tampered_record_fails_closed_without_stack_or_path(self):
        self.signal()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE events SET sha256='bad' WHERE event_key LIKE 'signal:%'")
        result = self.view()
        self.assertEqual(result["signals"], [])
        self.assertEqual(result["collector"]["status"], "unavailable")
        self.assertNotIn(str(self.path), json.dumps(result))

    def test_future_generated_candidate_rejected(self):
        self.signal(generated=self.at + timedelta(minutes=1))
        self.assertEqual(self.view()["collector"]["status"], "unavailable")

    def test_protocol_mismatch_not_silently_renamed(self):
        with sqlite3.connect(self.path) as db:
            row = json.loads(db.execute("SELECT payload FROM events WHERE seq=1").fetchone()[0])
            row["payload"]["implementation_sha256"]["brooks_failed_range.py"] = "other"
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
            db.execute("UPDATE events SET payload=?,sha256=? WHERE seq=1",
                       (payload, hashlib.sha256(payload.encode()).hexdigest()))
        self.assertEqual(self.view()["collector"]["status"], "unavailable")

    def test_real_ledger_writer_compatible(self):
        from brooks_failed_range_shadow import initialize, write_event
        path = Path(self.temp.name) / "real.sqlite3"
        header = {"policy": {"version": "failed-range-prospective-shadow-v2", "venue": "Coinbase", "symbol": "ETH-USD"},
                  "implementation_sha256": {"brooks_failed_range.py": RULE_SHA256}}
        initialize(path, header)
        at = datetime.now(timezone.utc)
        write_event(path, "heartbeat:real:now", "heartbeat", {"received_at": at.isoformat()})
        self.assertEqual(read_snapshot(path, at + timedelta(seconds=1))["collector"]["status"], "live")

    def test_dashboard_keeps_research_separate_from_ai_directions(self):
        import pandas as pd
        from web import server
        research = self.view()
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with ExitStack() as stack:
            for name, value in (("compute_4h_context", {}), ("fetch_klines", empty), ("compute_levels", {}),
                                ("fetch_fast_price", 2500.), ("compute_vwap", None)):
                stack.enter_context(patch.object(server.bot, name, return_value=value))
            stack.enter_context(patch.object(server, "websocket_price", return_value=2500.))
            stack.enter_context(patch.object(server, "read_json", side_effect=lambda name, fallback: fallback))
            stack.enter_context(patch.object(server, "research_snapshot", return_value=research))
            result = server.snapshot()
        self.assertEqual(result["research_signals"], research)
        self.assertEqual(result["symbol"], "ETHUSDC")
        self.assertIsNone(result["4h"]["direction"])
        self.assertIsNone(result["15m"]["direction"])

    @unittest.skipUnless(shutil.which("node"), "Node required for browser logic checks")
    def test_browser_rendering_and_sound_deduplication(self):
        result = subprocess.run(["node", "-e", BROWSER_CHECK, str(ROOT / "web/static/app.js")],
                                text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


BROWSER_CHECK = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(){return {textContent:'',className:'',style:{},children:[],attrs:{},append(x){this.children.push(x)},
 replaceChildren(...xs){this.children=xs},setAttribute(k,v){this.attrs[k]=v},addEventListener(){}}}
const document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element};
const context={document,window:{},Date,Set,JSON,console,location:{protocol:'https:',host:'test'},
 fetch:()=>new Promise(()=>{}),setInterval(){},setTimeout(){},WebSocket:class {close(){}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
let calls=0;playSignalAlarm=()=>{calls++;signalAlarm.node={stop(){},disconnect(){}};return true};researchSoundEnabled=true;
const base={collector:{status:'live',label:'采集中'},updated_at:new Date().toISOString(),signals:[]};
const row={id:'BRK-FR-15M-v1:test',direction:'LONG',status:'new',reason:'<img src=x onerror=alert(1)>',
 generated_at:base.updated_at,entry_not_before:new Date(Date.now()+120000).toISOString(),deadline:new Date(Date.now()+3600000).toISOString()};
renderResearch({...base,signals:[row]});
if(calls!==0)throw Error('historical replay sounded');
renderResearch({...base,signals:[row]});
if(calls!==0)throw Error('duplicate sounded');
const next={...row,id:'BRK-FR-15M-v1:next'};
renderResearch({...base,signals:[next,row]});
if(calls!==1)throw Error('new signal missing sound');
renderResearch({...base,signals:[next,row]});
if(calls!==1)throw Error('reconnect duplicate sounded');
renderResearch({...base,signals:[{...row,id:'ended',status:'ended'}]});
if(calls!==1)throw Error('ended signal sounded');
renderResearch({...base,collector:{status:'stale'},signals:[{...row,id:'stale'}]});
if(calls!==1)throw Error('stale signal sounded');
renderResearch({...base,updated_at:new Date(Date.now()-60000).toISOString(),signals:[{...row,id:'delayed'}]});
if(calls!==1)throw Error('stale browser payload sounded');
researchSoundEnabled=false;renderResearch({...base,signals:[{...row,id:'muted'}]});
if(calls!==1)throw Error('muted signal sounded');
const card=researchCard(row);
if(!card.className.includes('long'))throw Error('long not green');
if(!researchCard({...row,direction:'SHORT'}).className.includes('short'))throw Error('short not red');
if(!researchCard({...row,status:'ended'}).className.includes('flat'))throw Error('expired is still active');
if(!card.children.some(x=>x.textContent===row.reason))throw Error('text not escaped');
`,context);
console.log('browser research rendering, freshness and audio dedup passed');
"""


if __name__ == "__main__":
    unittest.main()
