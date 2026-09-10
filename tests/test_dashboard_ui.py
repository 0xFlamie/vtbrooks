from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IDs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        self.ids.extend(value for name, value in attrs if name == "id")


class TestDashboardUI(unittest.TestCase):
    def test_merged_cards_have_unique_ids_and_folded_sources(self):
        text = (ROOT / "web/static/index.html").read_text()
        parser = IDs()
        parser.feed(text)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for identity in ("support4", "support", "plan4", "plan15", "reasons4", "reasons15", "research-alert-log"):
            self.assertIn(identity, parser.ids)
        self.assertIn('<details class="panel section source-fold">', text)
        self.assertNotIn('<section class="metrics">', text)
        self.assertIn('<details class="sound-options">', text)
        self.assertIn('id="signal-alarm-countdown"', text)
        self.assertIn('aria-label="停止本轮响铃"', text)

    @unittest.skipUnless(shutil.which("node"), "Node required")
    def test_event_scope_audio_state_reasons_and_safe_text(self):
        result = subprocess.run(["node", "-e", CHECK, str(ROOT / "web/static/app.js")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


CHECK = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const ids=new Map();
function element(tag='div'){return {tag,textContent:'',className:'',style:{},dataset:{},children:[],attrs:{},
 append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},
 setAttribute(k,v){this.attrs[k]=v},addEventListener(){},closest(){return this},
 querySelector(q){return this.querySelectorAll(q)[0]},querySelectorAll(q){
  const out=[];for(const child of this.children){if(child.className?.split(' ').includes(q.slice(1)))out.push(child);out.push(...child.querySelectorAll(q))}return out;
 }}}
const document={getElementById(id){if(!ids.has(id))ids.set(id,element());return ids.get(id)},createElement:element};
const context={document,window:{},Date,Set,Map,JSON,URL,console,performance:{now:()=>0},location:{protocol:'https:',host:'test'},
 fetch:()=>new Promise(()=>{}),setInterval(){},setTimeout(){},WebSocket:class{close(){}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
(async()=>{
 const now=Date.now(),iso=ms=>new Date(ms).toISOString();
 const event=(id,ms)=>({id,name:id,release_at:iso(ms),metrics:[],scenarios:[],expectation_status:'missing'});
 const data={updated_at:iso(now),collector_status:'ok',events:[event('today',now+3600000),event('later-today',now+7200000),event('next-week',now+4*86400000)]};
 renderMacro(data);if($('macro-list').children.length!==1)throw Error('not nearest group');
 renderMacro({...data,events:[event('later',now+4*86400000)]});
 if($('macro-list').children.length!==0||!$('macro-note').textContent.includes('24小时内展开'))throw Error('distant event expanded');
 renderMacro({...data,events:[event('A',now+3600000),event('B',now+3600000)]});
 if($('macro-list').children.length!==2)throw Error('simultaneous event lost');
 renderDirection('4',{direction:'SHORT',confidence:70,summary:'<img onerror=x>',latest_reasons:['4-A','4-B'],plan:null});
 if($('reason4').textContent!=='<img onerror=x>'||$('reasons4').children.length!==2)throw Error('unsafe summary or reasons');
 if($('confidence4').textContent.includes('%'))throw Error('score falsely presented as probability');
 renderPlan('plan15',{plan:{entry:100,sl:99,tp1:101,tp2:102,rule:'<script>x</script>'}});
 if(!$('plan15').children.some(x=>x.textContent==='<script>x</script>'))throw Error('plan text not safe');
 renderJournal([{time:'2026-09-11T00:00:00Z',reasons:['4-A','4-B','15-C']}]);
 const cols=$('journal').children[0].children[1].children;
 if(cols[1].children.at(-1).children.map(x=>x.textContent).join()!=='15-C')throw Error('15m reasons mixed');
 renderJournal([{reasons:['uncertain-4','uncertain-15']}]);
 if($('journal').children[0].children[1].children.some(col=>col.children.at(-1).children.some(li=>li.textContent.includes('uncertain'))))throw Error('ambiguous legacy reasons assigned');
 window.AudioContext=class{constructor(){this.state='suspended'}async resume(){this.state='running';this.onstatechange?.()}};
 let sounds=0;researchTone=()=>{if(researchAudio?.state!=='running')return false;sounds++;return true};
 playSignalAlarm=()=>{sounds++;signalAlarm.node={stop(){},disconnect(){}};return true};
 await toggleResearchSound();if($('research-sound').attrs['aria-pressed']!=='true')throw Error('not enabled');
 researchAudio.state='suspended';researchAudio.onstatechange();
 if(!$('research-sound').textContent.includes('暂停')||$('research-sound').attrs['aria-pressed']!=='false')throw Error('suspended audio falsely enabled');
 await toggleResearchSound();if(researchAudio.state!=='running'||!researchSoundEnabled)throw Error('resume toggled off');
 const health={status:'live',label:'采集中'},base={collector:health,updated_at:iso(now),signals:[],families:[]};renderResearch(base);
 const signal={id:'A',family:'h2',direction:'LONG',status:'new',available_at:iso(now),entry_not_before:iso(now+120000)};
 const count=sounds;renderResearch({...base,signals:[signal]});
 if(sounds!==count+1||!researchAlertLog[0].includes('已提交播放'))throw Error('new signal not recorded');
 renderResearch({...base,signals:[signal]});if(sounds!==count+1)throw Error('duplicate sound');
 await toggleResearchSound();renderResearch({...base,signals:[{...signal,id:'B'}]});
 if(!researchAlertLog[0].includes('开关未开启'))throw Error('silent reason absent');
 researchAudio.state='closed';await testResearchSound();
 if(researchAudio.state!=='running'||researchSoundEnabled)throw Error('closed context not recreated or audition enabled alerts');
 renderResearch({...base,signals:[{...signal,id:'C',status:'history'}]});
 if(!researchAlertLog[0].includes('回看'))throw Error('historical alert explanation absent');
 if(!researchAlertBlock(signal,true,now-30000).includes('过期'))throw Error('stale allowed');
 if(!researchAlertBlock(signal,true,now+30000).includes('时钟'))throw Error('clock skew allowed');
 console.log('PASS dashboard scope, safe render, period separation, audio suspension/resume, alert audit');
})().catch(e=>{console.error(e);process.exitCode=1});
`,context);
"""


if __name__ == "__main__":
    unittest.main()
