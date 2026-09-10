from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import vt_vote_bot as bot

ROOT = Path(__file__).resolve().parents[1]


class TestSignalAlarm(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_global_alarm_lifecycle_and_direction_freshness(self):
        result = subprocess.run(["node", "-e", ALARM_CHECK, str(ROOT / "web/static/app.js")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_macro_veto_has_no_decorative_emoji_and_deduplicates_old_prefix(self):
        original = {"direction": "LONG", "confidence": 40, "reasons": ["🚨 宏观硬闸门:旧", "宏观硬闸门:旧2", "结构依据"]}
        with patch.object(bot, "active_macro_veto", return_value={"direction": "SHORT", "event": "数据", "reason": "核对"}):
            result, _ = bot.apply_macro_veto(dict(original), dict(original))
        self.assertEqual(result["direction"], "SHORT")
        self.assertEqual(result["confidence"], 70)
        self.assertTrue(result["reasons"][0].startswith("宏观硬闸门:"))
        self.assertFalse(any("🚨" in reason for reason in result["reasons"]))
        self.assertEqual(result["reasons"][1:], ["结构依据"])


ALARM_CHECK = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let now=Date.parse('2026-09-11T00:00:00Z');
class Clock extends Date{constructor(...args){super(...(args.length?args:[now]))}static now(){return now}}
const ids=new Map();
function element(){return {textContent:'',className:'',children:[],attrs:{},
 append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},
 setAttribute(k,v){this.attrs[k]=v},addEventListener(){},closest(){return this}}}
const document={getElementById(id){if(!ids.has(id))ids.set(id,element());return ids.get(id)},createElement:element};
class Audio{
 constructor(){this.state='running';this.currentTime=10;this.sampleRate=8000;this.nodes=[];this.destination={}}
 createBuffer(ch,length,rate){assert.equal(ch,1);assert.equal(rate,8000);return {data:new Float32Array(length),getChannelData(){return this.data}}}
 createBufferSource(){const source={stops:[],connect(){},disconnect(){this.disconnected=true},
  start(){if(this.failStart)throw Error('start failed');this.started=true},stop(t){this.stops.push(t)}};
  source.failStart=this.failStart;this.nodes.push(source);return source}
 async resume(){this.state='running';this.onstatechange?.()}
}
const context={document,window:{AudioContext:Audio},Date:Clock,Set,Map,JSON,console,assert,Audio,
 advance(ms){now+=ms},location:{protocol:'https:',host:'test'},fetch:()=>new Promise(()=>{}),
 setInterval(){},setTimeout(){},WebSocket:class{close(){}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
(async()=>{
 researchAudio=new Audio();researchAudio.onstatechange=syncResearchSound;researchSoundEnabled=true;
 const iso=()=>new Date().toISOString();
 const layer=(id,direction='LONG')=>({source:'latest_display_after_macro_veto',decision_id:id,
  decision_at:iso(),shown_at:iso(),direction});
 const snapshot=(a,b)=>({updated_at:iso(),'4h':a,'15m':b});
 let a=layer('a'),b=layer('b','SHORT');notifyDirections(snapshot(a,b));
 assert.equal(signalAlarm.until,0,'first snapshot must not replay');
 advance(1000);a=layer('a2');b=layer('b2','SHORT');notifyDirections(snapshot(a,b));
 const deadline=signalAlarm.until,first=signalAlarm.node;
 assert.equal(deadline-Date.now(),300000);assert.equal(first.loop,true);
 assert.equal(first.stops[0],310,'native audio cutoff must be scheduled');
 assert.equal(first.buffer.data.length,9600);assert(first.buffer.data.some(x=>Math.abs(x)>.01));
 assert.equal(signalAlarm.sources.size,2);assert.equal(researchAudio.nodes.length,1);
 assert.equal($('signal-alarm-stop').disabled,false);
 advance(60000);notifySignals([{family:'failed_range',direction:'LONG'}]);
 assert.equal(signalAlarm.until,deadline,'additional signals must not extend deadline');
 assert.equal(signalAlarm.sources.size,3);assert.equal(researchAudio.nodes.length,1);
 notifyDirections(snapshot(a,b));assert.equal(researchAudio.nodes.length,1,'duplicate no replay');
 stopSignalAlarm();assert(first.disconnected);assert.equal(first.stops.at(-1),undefined);
 assert.equal(signalAlarm.until,0);assert(researchSoundEnabled);assert($('signal-alarm-stop').disabled);
 notifyDirections(snapshot(a,b));assert.equal(signalAlarm.until,0,'manual stop survives duplicates');
 advance(1000);a=layer('a3');notifyDirections(snapshot(a,b));assert(signalAlarm.node);
 const deadline2=signalAlarm.until,pauseNode=signalAlarm.node;
 advance(90000);researchAudio.state='suspended';syncResearchSound();
 assert.equal(signalAlarm.node,null);assert(pauseNode.disconnected);assert.equal(signalAlarm.until,deadline2);
 researchAudio.state='running';syncResearchSound();
 assert.equal(signalAlarm.node.stops[0],220,'resume only remaining 210 seconds');
 assert.equal(signalAlarm.until,deadline2);const resumed=signalAlarm.node;
 advance(210001);tickSignalAlarm();assert(resumed.disconnected);assert.equal(signalAlarm.until,0);
 syncResearchSound();assert.equal(signalAlarm.node,null,'expired resume must not restart');
 advance(1000);notifySignals([{family:'h2',direction:'SHORT'}]);const nativeEnd=signalAlarm.node;
 nativeEnd.onended();assert.equal(signalAlarm.until,0,'native cutoff resets UI');
 advance(1000);notifySignals([{family:'wedge',direction:'SHORT'}]);
 await toggleResearchSound();assert(!researchSoundEnabled);assert.equal(signalAlarm.node,null);
 notifySignals([{family:'outside',direction:'LONG'}]);assert.equal(signalAlarm.until,0);
 assert(researchAlertLog[0].includes('开关未开启'));
 researchSoundEnabled=true;researchAudio.state='suspended';
 notifySignals([{family:'retest',direction:'LONG'}]);assert(signalAlarm.until);assert.equal(signalAlarm.node,null);
 advance(300001);researchAudio.state='running';syncResearchSound();assert.equal(signalAlarm.until,0);
 researchAudio.failStart=true;notifySignals([{family:'double_test',direction:'LONG'}]);
 assert.equal(signalAlarm.node,null);assert(researchAudio.nodes.at(-1).disconnected,'failed start must disconnect');
 stopSignalAlarm();researchAudio.failStart=false;

 directionAlarmStates.clear();let old=layer('old');assert.equal(directionSignal(old,'4h',Date.now()),null);
 advance(1000);assert.equal(directionSignal({...old,direction:'SHORT'},'4h',Date.now()),null,'same timestamp rejected');
 const flip={...old,direction:'SHORT',shown_at:iso()};
 assert.equal(directionSignal(flip,'4h',Date.now()).direction,'SHORT','macro display change uses shown time');
 assert.equal(directionSignal(old,'4h',Date.now()),null,'out of order rejected');
 advance(1000);assert.equal(directionSignal(layer('neutral',null),'4h',Date.now()),null);
 advance(1000);assert(directionSignal(layer('new'),'4h',Date.now()));
 advance(1000);assert.equal(directionSignal({...layer('stale'),decision_at:new Date(Date.now()-120001).toISOString()},'4h',Date.now()),null);
 advance(1000);assert.equal(directionSignal(layer('late'),'4h',Date.now()-20001),null);
 advance(1000);assert.equal(directionSignal({...layer('future'),decision_at:new Date(Date.now()+5001).toISOString()},'4h',Date.now()),null);
 advance(1000);assert.equal(directionSignal({...layer('legacy'),source:'legacy_journal'},'4h',Date.now()),null);
 advance(1000);assert.equal(directionSignal({...layer('invalid'),shown_at:'invalid'},'4h',Date.now()),null);
 advance(1000);assert(directionSignal(layer('fresh'),'4h',Date.now()));
 assert.equal(plainMacroText('🚨 宏观硬闸门:数据'),'宏观硬闸门:数据');
 renderDirection('4',{summary:'🚨 宏观硬闸门 <img onerror=x>',latest_reasons:['🚨 宏观硬闸门:旧'],change:'🚨 宏观硬闸门:变动'});
 assert.equal($('reason4').textContent,'宏观硬闸门 <img onerror=x>');
 assert.equal($('reasons4').children[0].textContent,'宏观硬闸门:旧');
 assert(!$('change4').textContent.includes('🚨'));
 console.log('PASS global 5min loop, native cutoff, manual stop, merge, pause/resume, dedup, freshness, safe legacy text');
})().catch(e=>{console.error(e);process.exitCode=1});
`,context);
"""


if __name__ == "__main__":
    unittest.main()
