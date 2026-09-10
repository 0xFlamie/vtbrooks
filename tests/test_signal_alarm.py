from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import vt_vote_bot as bot

ROOT = Path(__file__).resolve().parents[1]


class TestSignalAlarm(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_brooks_alarm_lifecycle_and_silent_direction_cards(self):
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
function element(){return {textContent:'',className:'',children:[],attrs:{},dataset:{},style:{},
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
const context={document,window:{AudioContext:Audio},Date:Clock,Set,Map,JSON,console,assert,Audio,process,
 advance(ms){now+=ms},location:{protocol:'https:',host:'test'},fetch:()=>new Promise(()=>{}),
 setInterval(){},setTimeout(){},WebSocket:class{close(){}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
(async()=>{
 researchAudio=new Audio();researchAudio.onstatechange=syncResearchSound;researchSoundEnabled=true;
 const iso=()=>new Date().toISOString();
 const layer=(id,family='h2',direction='LONG')=>({id,family,direction,status:'new',available_at:iso(),
  generated_at:iso(),entry_not_before:new Date(Date.now()+120000).toISOString()});
 const snapshot=(...signals)=>({updated_at:iso(),collector:{status:'live'},signals});
 let a=layer('a'),b=layer('b','retest','SHORT');renderResearch(snapshot(a,b));
 assert.equal(signalAlarm.until,0,'first snapshot must not replay');
 advance(1000);a=layer('a2');b=layer('b2','retest','SHORT');renderResearch(snapshot(a,b));
 const deadline=signalAlarm.until,first=signalAlarm.node;
 assert.equal(deadline-Date.now(),300000);assert.equal(first.loop,true);
 assert.equal(first.stops[0],310,'native audio cutoff must be scheduled');
 assert.equal(first.buffer.data.length,9600);assert(first.buffer.data.some(x=>Math.abs(x)>.01));
 assert.equal(signalAlarm.sources.size,2);assert.equal(researchAudio.nodes.length,1);
 assert.equal($('signal-alarm-stop').disabled,false);
 assert.equal($('signal-alarm-stop').hidden,false);assert.equal($('signal-alarm-countdown').hidden,false);
 assert.equal($('signal-alarm-countdown').textContent,'5:00');
 advance(60000);notifySignals([{family:'failed_range',direction:'LONG'}]);
 assert.equal($('signal-alarm-countdown').textContent,'4:00');
 assert.equal(signalAlarm.until,deadline,'additional signals must not extend deadline');
 assert.equal(signalAlarm.sources.size,3);assert.equal(researchAudio.nodes.length,1);
 renderResearch(snapshot(a,b));assert.equal(researchAudio.nodes.length,1,'duplicate no replay');
 stopSignalAlarm();assert(first.disconnected);assert.equal(first.stops.at(-1),undefined);
 assert.equal(signalAlarm.until,0);assert(researchSoundEnabled);assert($('signal-alarm-stop').disabled);
 assert($('signal-alarm-stop').hidden);assert($('signal-alarm-countdown').hidden);
 renderResearch(snapshot(a,b));assert.equal(signalAlarm.until,0,'manual stop survives duplicates');
 advance(1000);a=layer('a3');renderResearch(snapshot(a,b));assert(signalAlarm.node);
 const deadline2=signalAlarm.until,pauseNode=signalAlarm.node;
 advance(90000);researchAudio.state='suspended';syncResearchSound();
 assert.equal(signalAlarm.node,null);assert(pauseNode.disconnected);assert.equal(signalAlarm.until,deadline2);
 assert.equal($('research-sound').attrs['aria-label'],'恢复Brooks形态声音');
 researchAudio.state='running';syncResearchSound();
 assert.equal(signalAlarm.node.stops[0],220,'resume only remaining 210 seconds');
 assert.equal(signalAlarm.until,deadline2);const resumed=signalAlarm.node;
 assert.equal($('research-sound').textContent,'声音开');
 assert.equal($('research-sound').attrs['aria-label'],'关闭Brooks形态声音');
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

 const nodes=researchAudio.nodes.length;
 for(const direction of ['LONG','SHORT',null,'LONG']){
  advance(1000);
  const judgment={source:'latest_display_after_macro_veto',decision_id:iso(),decision_at:iso(),shown_at:iso(),direction};
  render({updated_at:iso(),'4h':judgment,'15m':judgment,research_signals:snapshot()});
  assert.equal(signalAlarm.until,0,'direction changes must not ring');
 }
 assert.equal(researchAudio.nodes.length,nodes);
 advance(1000);renderResearch(snapshot(layer('after-directions','wedge')));
 assert(signalAlarm.node,'Brooks still rings after silent direction changes');stopSignalAlarm();
 assert.equal(plainMacroText('🚨 宏观硬闸门:数据'),'宏观硬闸门:数据');
 renderDirection('4',{summary:'🚨 宏观硬闸门 <img onerror=x>',latest_reasons:['🚨 宏观硬闸门:旧'],change:'🚨 宏观硬闸门:变动'});
 assert.equal($('reason4').textContent,'宏观硬闸门 <img onerror=x>');
 assert.equal($('reasons4').children[0].textContent,'宏观硬闸门:旧');
 assert(!$('change4').textContent.includes('🚨'));
 console.log('PASS Brooks 5min loop, native cutoff, manual stop, merge, pause/resume, dedup, silent directions, safe legacy text');
})().catch(e=>{console.error(e);process.exitCode=1});
`,context);
"""


if __name__ == "__main__":
    unittest.main()
