from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import shutil
import subprocess
from unittest.mock import patch

import pandas as pd

from web.research_signals import RULE_SHA256
from web.seven_signals import CONTEXT_URL, FAMILIES, SevenReader, timely_prefix


class TestSevenSignals(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.reader = SevenReader()
        self.at = pd.Timestamp("2026-09-10T08:02Z")
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE events(seq INTEGER PRIMARY KEY,event_key TEXT UNIQUE,payload TEXT,sha256 TEXT)")
        self.put("protocol", "protocol", {"policy": {"version": "failed-range-prospective-shadow-v2",
                 "venue": "Coinbase", "symbol": "ETH-USD"},
                 "implementation_sha256": {"brooks_failed_range.py": RULE_SHA256}})
        self.heartbeat()

    def put(self, key, kind, payload):
        with sqlite3.connect(self.path) as db:
            tail = db.execute("SELECT sha256 FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            event = {"key": key, "kind": kind, "payload": payload, "recorded_at": self.at.isoformat(),
                     "previous_sha256": tail[0] if tail else None}
            text = json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
            db.execute("INSERT INTO events(event_key,payload,sha256) VALUES (?,?,?)",
                       (key, text, hashlib.sha256(text.encode()).hexdigest()))

    def heartbeat(self):
        self.put("heartbeat:test:" + self.at.isoformat(), "heartbeat", {"received_at": self.at.isoformat()})

    def seed_double(self, context_only_first=False):
        stamp = self.at.floor("15min")
        begin = stamp - pd.Timedelta(minutes=128 * 15)
        frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.},
                             index=pd.date_range(begin, self.at, freq="min", inclusive="left"))
        anchor = stamp - pd.Timedelta(hours=1)
        frame.loc[anchor:anchor + pd.Timedelta(minutes=14), "low"] = 98.
        frame.loc[stamp] = [98., 98.2, 98., 98.1, 1.]
        frame.loc[stamp + pd.Timedelta(minutes=1)] = [98.3, 98.5, 98.3, 98.4, 1.]
        history = frame.loc[frame.index < stamp + pd.Timedelta(minutes=int(context_only_first))]
        for i in range(0, len(history), 240):
            part = history.iloc[i:i + 240]
            payload = [[int(t.timestamp()), r.low, r.high, r.open, r.close, r.volume] for t, r in part.iterrows()]
            self.put(f"context:{i}", "context", {"endpoint": CONTEXT_URL, "live": False,
                     "params": {"start": part.index[0].isoformat(), "end": (part.index[-1] + pd.Timedelta(minutes=1)).isoformat(), "granularity": 60},
                     "received_at": self.at.isoformat(), "payload": payload})
        for t, row in frame.loc[frame.index >= stamp + pd.Timedelta(minutes=int(context_only_first))].iterrows():
            self.put("minute:" + t.isoformat(), "minute", {"stamp": t.isoformat(), **row.to_dict(),
                     "live": True, "received_at": (t + pd.Timedelta(minutes=1)).isoformat(), "trades": []})
        return frame

    def view(self):
        return self.reader.snapshot(self.path, self.at)

    def test_always_seven_distinct_families_and_new_three_year_rates(self):
        value = self.view()
        self.assertEqual([f["family"] for f in value["families"]], [f[0] for f in FAMILIES])
        self.assertEqual(len(set(f["strategy_id"] for f in value["families"])), 7)
        self.assertEqual(value["signals"], [])
        self.assertEqual(value["families"][3]["rate"], "65.7%–65.9%")
        self.assertIn("不是盈利胜率", value["history_note"])
        self.assertTrue(all(f["collector"]["status"] == "unavailable" for f in value["families"] if f["family"] != "failed_range"))

    def test_real_engine_live_double_is_new_readonly_and_no_trade_payload(self):
        self.seed_double()
        before = self.path.read_bytes()
        value = self.view()
        signal = next(s for s in value["signals"] if s["family"] == "double_test")
        self.assertEqual(signal["status"], "new")
        self.assertEqual(signal["direction"], "LONG")
        self.assertEqual(signal["observed_price"], 98.4)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn('"trades"', json.dumps(value))
        self.assertNotIn("entry_price", signal)

    def test_no_rescan_until_new_minute_and_generation_is_stable(self):
        self.seed_double()
        first = self.view()
        with patch("web.seven_signals.scan_six", side_effect=AssertionError("unnecessary rescan")):
            second = self.view()
        self.assertEqual(first, second)

    def test_rest_trigger_minute_cannot_become_live_alert(self):
        self.seed_double(context_only_first=True)
        signals = [s for s in self.view()["signals"] if s["family"] == "double_test"]
        self.assertTrue(signals)
        self.assertTrue(all(s["status"] == "history" for s in signals))

    def test_fresh_heartbeat_does_not_hide_stale_minute_cache(self):
        self.seed_double()
        self.assertTrue(self.view()["signals"])
        self.at += pd.Timedelta(minutes=3)
        self.heartbeat()
        value = self.view()
        self.assertTrue(all(s["status"] == "unverified" for s in value["signals"]))
        self.assertEqual(value["families"][0]["collector"]["status"], "unavailable")

    def test_tampered_minute_fails_closed(self):
        self.seed_double()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE events SET sha256='bad' WHERE event_key LIKE 'minute:%'")
        self.assertEqual(self.view()["signals"], [])

    def test_missing_database_not_created(self):
        path = Path(self.temp.name) / "missing.sqlite3"
        value = self.reader.snapshot(path, self.at)
        self.assertFalse(path.exists())
        self.assertEqual(len(value["families"]), 7)
        self.assertEqual(value["signals"], [])

    def test_prefix_rejects_gap_future_receive_or_late_detection(self):
        signal = {"bar_start": (self.at - pd.Timedelta(minutes=1)).isoformat(), "available_at": self.at.isoformat()}
        stamp = signal["bar_start"]
        self.assertFalse(timely_prefix(signal, {}, self.at))
        self.assertFalse(timely_prefix(signal, {stamp: {"live": True, "received_at": (self.at + timedelta(seconds=1)).isoformat()}}, self.at))
        self.assertFalse(timely_prefix(signal, {stamp: {"live": True, "received_at": self.at.isoformat()}}, self.at + timedelta(minutes=2)))

    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_seven_tiles_independent_flashes_and_batched_sound(self):
        script = Path(__file__).resolve().parents[1] / "web/static/app.js"
        result = subprocess.run(["node", "-e", BROWSER_CHECK, str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


BROWSER_CHECK = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(){return {textContent:'',className:'',style:{},children:[],attrs:{},append(x){this.children.push(x)},
replaceChildren(...xs){this.children=xs},setAttribute(k,v){this.attrs[k]=v},addEventListener(){}}}
const document={getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id)},createElement:element};
const context={document,window:{},Date,Set,Map,JSON,console,location:{protocol:'https:',host:'test'},
fetch:()=>new Promise(()=>{}),setInterval(){},setTimeout(){},WebSocket:class{close(){}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
let calls=0;researchTone=()=>{calls++;return true};researchSoundEnabled=true;
const now=new Date().toISOString(),health={status:'live',label:'采集中'};
const families=['h2','retest','outside','failed_range','second_leg','double_test','wedge'].map(f=>({family:f,name:f,collector:health,signals:[]}));
const base={collector:health,updated_at:now,families,signals:[]};
renderResearch(base);
if($('research-list').children.length!==7)throw Error('not seven tiles');
const mk=(family,direction)=>({id:family+':1',family,direction,status:'new',available_at:now,generated_at:now,
entry_not_before:new Date(Date.now()+120000).toISOString(),deadline:new Date(Date.now()+3600000).toISOString(),reason:'<img onerror=x>'});
const a=mk('h2','LONG'),b=mk('wedge','SHORT');
const value={...base,signals:[a,b],families:families.map(f=>({...f,signals:[a,b].filter(s=>s.family===f.family)}))};
renderResearch(value);
const cards=$('research-list').children;
if(calls!==1)throw Error('concurrent alerts must share one tone');
if(!cards[0].className.includes('long')||!cards[6].className.includes('short'))throw Error('direction colors');
if(cards.filter(c=>c.className.includes('research-flash')).length!==2)throw Error('unrelated tile flashed');
renderResearch(value);if(calls!==1)throw Error('duplicate sounded');
researchInitialized=false;researchSeen.clear();researchFlashes.clear();renderResearch(value);
if(calls!==1||$('research-list').children.some(c=>c.className.includes('research-flash')))throw Error('history replay');
const stale={...a,id:'stale',status:'unverified'};renderResearch({...base,signals:[stale]});
if(calls!==1)throw Error('unverified alert sounded');
const tile=researchTile({...families[0],signals:[a]});
if(!tile.children.at(-1).children.some(c=>c.textContent==='<img onerror=x>'))throw Error('unsafe text or missing reason');
const conflict=researchTile({...families[0],signals:[a,{...a,id:'opposite',direction:'SHORT'}]});
if(!conflict.className.includes('flat')||!conflict.children.some(c=>c.textContent==='多空同现'))throw Error('conflicting directions silently selected');
`,context);
"""


if __name__ == "__main__":
    unittest.main()
