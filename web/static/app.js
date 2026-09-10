const $ = (id) => document.getElementById(id);
const plainMacroText=value=>typeof value==="string"?value.replace(/🚨\uFE0F?\s*(?=宏观硬闸门)/g,""):value;
const cnDir = (v) => ({bull_leg:"多头腿",bear_leg:"空头腿",range:"震荡/无连续方向"}[v] || v || "—");
const fmt = (v) => typeof v === "number" ? v.toFixed(2) : (v ?? "—");
const dirText = (v) => v === "LONG" ? "做多" : v === "SHORT" ? "做空" : "观望";
const dirClass = (v) => v === "LONG" ? "long" : v === "SHORT" ? "short" : "flat";
const structureText = (trend, direction) => {
  if (trend === "range" && direction === "LONG") return "震荡中偏多";
  if (trend === "range" && direction === "SHORT") return "震荡中偏空";
  if (trend === "bull_leg" && direction === "SHORT") return "上升趋势中转弱";
  if (trend === "bear_leg" && direction === "LONG") return "下降趋势中反弹";
  return cnDir(trend);
};
function putModules(id, rows){
  $(id).replaceChildren(...rows.map(([k,t,v])=>{
    const card=document.createElement("article");card.className="module-card";
    for(const [tag,value] of [["span",k],["b",t],["small",v]]){const el=document.createElement(tag);el.textContent=value??"—";card.append(el);}
    return card;
  }));
}
function evidenceRows(brooks){
  const evidence=brooks.evidence||{}, quality=evidence.quality||{}, structure=evidence.structure||{};
  const labels={ok:"数据可用",degraded:"需留意数据质量",invalid:"暂不可确认"};
  const points=(evidence.swings||[]).map(p=>`${p.kind==="high"?"高":"低"} $${fmt(p.price)}`).join(" · ");
  return [
    ["数据质量",labels[quality.status]||"等待检查",quality.message||"质量信息随快照更新"],
    ["已确认摆动",structure.description||"等待结构事实",`${points||"确认摆动点不足"}${structure.invalid!=null?` · 结构检验位 $${fmt(structure.invalid)}`:""} · 右侧两根收线确认，不是胜率预测`]
  ];
}
function numberedReasons(id,reasons){
  $(id).replaceChildren(...(Array.isArray(reasons)&&reasons.length?reasons:["本轮未提供分项依据"]).map(text=>macroText("li",text)));
}
function renderDirection(id,layer){
  const direction=layer.direction;
  $("direction"+id).textContent=dirText(direction);$("direction"+id).className=dirClass(direction);
  $("direction"+id).closest(".direction").className=`direction panel ${dirClass(direction)}`;
  $("confidence"+id).textContent=layer.confidence==null||layer.confidence<0?"暂无评分":`参考评分 ${layer.confidence}`;
  $("confidence"+id).title="模型自评或规则调整值，不是经回测校准的盈利概率";
  $("reason"+id).textContent=plainMacroText(layer.summary)||"本轮未提供一句话判断";
  numberedReasons("reasons"+id,layer.latest_reasons);
  $("decision-time"+id).textContent=layer.decision_at?`技术判断 ${macroTime(layer.decision_at)} · 不是逐笔重判`:"判断时间未记录 · 请留意旧观点";
  if(layer.shown_at&&Date.now()-Date.parse(layer.shown_at)>600000)$("decision-time"+id).textContent+=" · 展示超过10分钟未更新";
  $("change"+id).textContent=plainMacroText([layer.change,layer.previous_summary?`上次：${layer.previous_summary}`:""].filter(Boolean).join(" · "));
  if(layer.source==="legacy_journal")$("change"+id).textContent="历史记录回退，非最新逐次判决";
  renderPlan("plan"+id,layer);
}
function renderPlan(id,layer){
  const box=$(id);box.replaceChildren(macroText("b","结构进出场测算 · 非触发信号"));
  const p=layer.plan;
  if(!p){box.append(macroText("small",layer.plan_note||"没有可用入场计划，等待同向结构确认"));return;}
  const grid=macroText("div","","plan-levels");
  for(const [label,value] of [["参考位",p.entry],["失效 / 止损",p.sl],["目标一",p.tp1],["目标二",p.tp2]]){
    const cell=macroText("div","");cell.append(macroText("small",label),macroText("span",`$${fmt(value)}`));grid.append(cell);
  }
  box.append(grid,macroText("small",p.rule||"等待确认条件"),macroText("small","按本周期K线计算；不是实盘成交价或已经验证的止损。"));
}
let journalSignature="";
function renderJournal(entries){
  const signature=JSON.stringify(entries||[]);if(signature===journalSignature)return;journalSignature=signature;
  const cards=(entries||[]).slice().reverse().map(x=>{
    const card=macroText("article","","journal-entry"),head=macroText("div","","journal-head");
    const time=String(x.time||"时间未记录");
    head.append(macroText("time",/[zZ]|[+-]\d\d:\d\d$/.test(time)?macroTime(time):time.replace("T"," ").slice(0,19)+" · 原记录时间"));
    head.append(macroText("span",`4小时 ${dirText(x.dir4h)} / 15分钟 ${dirText(x.dir15m)}`));card.append(head);
    const columns=macroText("div","","journal-columns");
    for(const tf of ["4h","15m"]){
      const old=x.reasons||[],part=macroText("section",""),reasons=x["reasons"+tf]||(old.length===3?(tf==="4h"?old.slice(0,2):old.slice(2,3)):[]);
      part.append(macroText("b",tf==="4h"?"4小时依据":"15分钟依据"));
      if(x["summary"+tf])part.append(macroText("p",x["summary"+tf]));
      const list=macroText("ol","");for(const reason of reasons.length?reasons:["旧记录未保留该周期依据"])list.append(macroText("li",reason));part.append(list);columns.append(part);
    }
    card.append(columns);return card;
  });
  const older=macroText("details","","context-fold");older.append(macroText("summary",`更早记录（${Math.max(cards.length-2,0)}）`),...cards.slice(2));
  $("journal").replaceChildren(...(cards.length?cards.slice(0,2):[macroText("p","暂无判决记录")]),...(cards.length>2?[older]:[]));
}
const researchSeen=new Set();
const researchFlashes=new Map();
let researchInitialized=false, researchAudio=null, researchSoundEnabled=false, researchLastUpdate=0, researchSignature="";
const researchTime=v=>new Date(v).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false});
const SIGNAL_ALARM_MS=5*60*1000;
const signalAlarm={until:0,node:null,sources:new Set(),message:"等待新信号"};
function dropAlarmAudio(){
  const source=signalAlarm.node;signalAlarm.node=null;
  if(source){source.onended=null;source.stop();source.disconnect();}
}
function stopSignalAlarm(message="已手动停止本轮；下个新信号仍会提醒"){
  dropAlarmAudio();signalAlarm.until=0;signalAlarm.sources.clear();signalAlarm.message=message;showSignalAlarm();
}
function playSignalAlarm(){
  if(!researchAudio||researchAudio.state!=="running"||signalAlarm.until<=Date.now())return false;
  let source=null,started=false;
  try{
    const rate=researchAudio.sampleRate,buffer=researchAudio.createBuffer(1,Math.ceil(rate*1.2),rate),data=buffer.getChannelData(0);
    for(let i=0;i<data.length;i++){
      const t=i/rate,part=t%.4,hz=Math.floor(t/.4)===1?880:660;
      data[i]=part<.28?.16*Math.min(part/.02,1)*Math.exp(-12*part)*Math.sin(2*Math.PI*hz*part):0;
    }
    source=researchAudio.createBufferSource();source.buffer=buffer;source.loop=true;source.connect(researchAudio.destination);
    source.start();started=true;source.stop(researchAudio.currentTime+Math.max(0,(signalAlarm.until-Date.now())/1000));
    source.onended=()=>{if(signalAlarm.node===source)stopSignalAlarm("本轮5分钟响铃结束；等待新信号")};
    signalAlarm.node=source;return true;
  }catch(_){
    if(source){source.disconnect();if(started)source.stop();}
    signalAlarm.message="音频播放失败，请检查权限并重新开启声音";return false;
  }
}
function showSignalAlarm(){
  const active=signalAlarm.until>Date.now(),seconds=Math.max(0,Math.ceil((signalAlarm.until-Date.now())/1000));
  const status=$("signal-alarm-status"),stop=$("signal-alarm-stop");
  if(stop)stop.disabled=!active;
  if(status)status.textContent=active?`${[...signalAlarm.sources].join(" / ")} · ${signalAlarm.node?"循环响铃":"声音暂停 / 未能播放"} ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")} · 提醒不代表当前仍可入场`:signalAlarm.message;
}
function tickSignalAlarm(){
  if(signalAlarm.until&&Date.now()>=signalAlarm.until){stopSignalAlarm("本轮5分钟响铃结束；等待新信号");return;}
  if(signalAlarm.until&&researchAudio?.state!=="running")dropAlarmAudio();
  showSignalAlarm();
}
function startSignalAlarm(labels){
  if(!researchSoundEnabled)return false;
  if(signalAlarm.until&&Date.now()>=signalAlarm.until)stopSignalAlarm("本轮结束");
  if(signalAlarm.until){for(const label of labels)signalAlarm.sources.add(label);showSignalAlarm();return !!signalAlarm.node;}
  signalAlarm.until=Date.now()+SIGNAL_ALARM_MS;signalAlarm.sources=new Set(labels);
  const played=playSignalAlarm();showSignalAlarm();return played;
}
const signalNames={"4h":"4小时方向","15m":"15分钟方向",h2:"二次入场",retest:"突破回触",outside:"外包K",failed_range:"区间假突破",second_leg:"第二腿",double_test:"双顶双底",wedge:"楔形"};
const signalLabel=signal=>`${signalNames[signal.family]||"Brooks"} · ${signal.direction==="LONG"?"偏多":"偏空"}`;
function notifySignals(signals){
  if(!signals.length)return;
  const played=startSignalAlarm(signals.map(signalLabel));
  for(const signal of signals)logResearchAlert(signal,played?"已提交播放 / 合并本轮5分钟提醒（设备是否发声无法确认）":researchSoundEnabled?"浏览器暂停或播放失败，请恢复剩余提醒":"声音开关未开启，未播放");
}
const directionAlarmStates=new Map();
function directionSignal(layer,tf,serverAt){
  if(layer?.source!=="latest_display_after_macro_veto"||!layer.decision_id)return null;
  const shown=Date.parse(layer.shown_at),old=directionAlarmStates.get(tf),key=`${layer.decision_id}:${layer.direction}`;
  if(!Number.isFinite(shown)||(old&&shown<=old.shown))return null;
  directionAlarmStates.set(tf,{key,shown,id:layer.decision_id});
  if(!old||old.key===key||!["LONG","SHORT"].includes(layer.direction))return null;
  const signal={family:tf,direction:layer.direction};
  const eventAt=old.id===layer.decision_id?shown:Date.parse(layer.decision_at),age=Date.now()-eventAt,lag=Date.now()-serverAt;
  if(!(age>=-5000&&age<=120000&&lag>=-5000&&lag<=20000)){
    logResearchAlert(signal,"方向判决到达过期或时钟异常，不追旧信号");return null;
  }
  return signal;
}
function notifyDirections(snapshot){
  const at=Date.parse(snapshot.updated_at);
  notifySignals(["4h","15m"].map(tf=>directionSignal(snapshot[tf],tf,at)).filter(Boolean));
}
function researchTone(preview=false){
  if(!researchAudio||researchAudio.state!=="running")return false;
  const start=researchAudio.currentTime;
  for(const [delay,hz] of (preview?[[0,660]]:[[0,660],[.32,880],[.64,660]])){
    const oscillator=researchAudio.createOscillator(), gain=researchAudio.createGain();
    oscillator.type="sine";oscillator.frequency.value=hz;gain.gain.setValueAtTime(0,start+delay);
    gain.gain.linearRampToValueAtTime(.18,start+delay+.02);gain.gain.exponentialRampToValueAtTime(.001,start+delay+.26);
    oscillator.connect(gain);gain.connect(researchAudio.destination);oscillator.start(start+delay);oscillator.stop(start+delay+.28);
  }
  return true;
}
function syncResearchSound(){
  const running=researchSoundEnabled&&researchAudio?.state==="running",button=$("research-sound");
  button.textContent=running?"全站声音已开启":researchSoundEnabled?"声音暂停 · 点击恢复":"开启全站信号声音";
  button.className=running?"pill":"pill muted";button.setAttribute("aria-pressed",String(running));
  $("research-sound-note").textContent=running?"4h / 15m / Brooks新信号循环响5分钟；同轮不续时。刷新需重开，后台/静音可能漏提醒。":researchSoundEnabled?"浏览器暂停音频；点击恢复本轮剩余时间，已过期不补响。":"声音未开启；卡片仍会更新。先试听，再开启全站提醒。";
  tickSignalAlarm();
  if(signalAlarm.until&&running&&!signalAlarm.node)playSignalAlarm();
  showSignalAlarm();
}
async function prepareResearchAudio(){
  const Audio=window.AudioContext||window.webkitAudioContext;
  if(!Audio)throw new Error("unsupported");
  if(!researchAudio||researchAudio.state==="closed"){researchAudio=new Audio();researchAudio.onstatechange=syncResearchSound;}
  await researchAudio.resume();
  if(researchAudio.state!=="running")throw new Error("suspended");
}
async function toggleResearchSound(){
  if(researchSoundEnabled&&researchAudio?.state==="running"){researchSoundEnabled=false;stopSignalAlarm("全站声音已关闭");syncResearchSound();return;}
  try{
    await prepareResearchAudio();researchSoundEnabled=true;syncResearchSound();if(!signalAlarm.until)researchTone(true);
  }catch(_){researchSoundEnabled=false;syncResearchSound();$("research-sound-note").textContent="声音未能开启，请检查浏览器声音权限；仍可查看卡片。";}
}
async function testResearchSound(){
  if(signalAlarm.until>Date.now()){showSignalAlarm();return;}
  try{await prepareResearchAudio();researchTone();$("research-sound-note").textContent="已请求播放三声试听；请自己确认能否听到。试听不改变提醒开关。";}
  catch(_){$("research-sound-note").textContent="试听失败，请检查浏览器权限和设备声音。";}
}
const researchAlertLog=[];
function logResearchAlert(signal,outcome){
  const label=`${researchTime(new Date().toISOString())} · ${signalLabel(signal)} · ${outcome}`;
  researchAlertLog.unshift(label);researchAlertLog.splice(20);
  $("research-alert-log")?.replaceChildren(...researchAlertLog.map(text=>macroText("li",text)));
}
function researchAlertBlock(signal,live,serverAt){
  if(!researchInitialized)return "初次载入，仅展示，不重播";
  if(!live||signal.status==="unverified")return "数据未就绪或中断，不播放";
  if(signal.status!=="new")return "回看 / 观察中 / 已结束，不作为新提醒";
  if(!(serverAt<Date.parse(signal.entry_not_before)&&Date.now()<Date.parse(signal.entry_not_before)))return "到达时已过新提醒窗口，不追旧信号";
  if(!(Date.now()-serverAt>=-5000&&Date.now()-serverAt<=20000))return "快照过期或设备时钟偏差，不播放";
  return null;
}
function researchCard(signal){
  const active=signal.status==="new"||signal.status==="observing", card=document.createElement("article");
  card.className=`direction panel research-event ${active?dirClass(signal.direction):"flat"}`;
  const labels={new:"新出现 · 等待研究参考时刻",observing:"观察中 · 参考时刻已过，不作追单提示",ended:"观察窗口已结束",unverified:"行情中断 · 暂不能确认"};
  const fields=[["div",`${signal.id}`,"eyebrow"],["strong",`${signal.direction==="LONG"?"偏多":"偏空"} · 研究候选`],
    ["div",labels[signal.status]||"状态未知","pill muted"],["p",signal.reason],
    ["p",`触发时价格 $${fmt(signal.observed_price)} · 区间 $${fmt(signal.range_low)}–$${fmt(signal.range_high)} · ATR $${fmt(signal.atr)}`],
    ["p",`信号结构参考 $${fmt(signal.structure_reference)}（不是已验证止损）；回到区间外需重看假突破思路`],
    ["p",`生成 ${researchTime(signal.generated_at)} · 研究参考 ${researchTime(signal.entry_not_before)} · 窗口截止 ${researchTime(signal.deadline)}（北京时间）`],
    ["small",signal.result]];
  for(const [tag,text,cls] of fields){const el=document.createElement(tag);el.textContent=text;if(cls)el.className=cls;card.append(el);}
  return card;
}
function researchTile(family){
  const signal=family.signals?.[0], healthy=family.collector?.status==="live";
  const simultaneous=(family.signals||[]).filter(s=>s.available_at===signal?.available_at&&["new","observing"].includes(s.status));
  const conflict=healthy&&new Set(simultaneous.map(s=>s.direction)).size>1;
  const active=healthy&&!conflict&&signal&&["new","observing"].includes(signal.status);
  const card=document.createElement("article");
  const flashing=healthy&&simultaneous.some(s=>s.status==="new"&&(researchFlashes.get(s.id)||0)>Date.now());
  card.className=`research-tile ${active?dirClass(signal.direction):"flat"}${flashing?" research-flash":""}`;
  card.setAttribute("data-family",family.family);
  const labels={new:"新信号",observing:"观察中 · 不追单",history:"回看 · 不响铃",ended:"已结束",unverified:"数据中断"};
  const direction=signal?(signal.direction==="LONG"?"偏多":"偏空"):"等待形态";
  const fields=[["div",family.name,"tile-name"],["strong",conflict?"多空同现":active?direction:signal?`上次${direction}`:"—"],
    ["small",conflict?"同分钟冲突 · 不合并方向":!healthy?(family.collector?.label||"连接中"):signal?(labels[signal.status]||"状态未知"):"等待形态"],
    ["div",signal?`触发 $${fmt(signal.observed_price)}`:"触发价 —","tile-price"],
    ["small",signal?`ATR $${fmt(signal.atr)}`:"ATR —"],
    ["small",signal?researchTime(signal.available_at):"15M · 盘中观察"],
    ["small",`三年空间率 ${family.rate||"—"}`,"tile-rate"]];
  for(const [tag,text,cls] of fields){const el=document.createElement(tag);el.textContent=text;if(cls)el.className=cls;card.append(el);}
  const details=document.createElement("details"), summary=document.createElement("summary");summary.textContent="依据 / 结构位";details.append(summary);
  for(const text of [family.strategy_id,signal?.reason||"暂无候选，不代表AI观望判决",
    signal?`结构参考 $${fmt(signal.structure_reference)} · 非验证止损`:"等待真实信号后显示点位",
    `历史样本 ${family.sample??"—"} · 空间率非盈利胜率`,signal?.result||"研究提醒，不自动开单"]){
    const p=document.createElement("p");p.textContent=text;details.append(p);
  }
  if(conflict)for(const s of simultaneous){const p=document.createElement("p");p.textContent=`${s.direction==="LONG"?"偏多":"偏空"} · 触发 $${fmt(s.observed_price)} · 结构 $${fmt(s.structure_reference)}`;details.append(p);}
  card.append(details);return card;
}
function renderResearch(data){
  if(!$("research-list"))return;
  const live=data?.collector?.status==="live", signals=data?.signals||[], serverAt=Date.parse(data?.updated_at);
  const incoming=[];
  for(const signal of signals){
    if(!researchSeen.has(signal.id)){
      const conflict=signals.some(other=>other.family===signal.family&&other.available_at===signal.available_at&&other.direction!==signal.direction&&other.status==="new");
      const blocked=researchAlertBlock(signal,live,serverAt)||(conflict?"同类同刻多空冲突，不播放":null);
      if(blocked)logResearchAlert(signal,blocked);
      else{incoming.push(signal);researchFlashes.set(signal.id,Date.now()+8000);}
    }
    researchSeen.add(signal.id);
  }
  while(researchSeen.size>1000)researchSeen.delete(researchSeen.values().next().value);
  for(const [id,until] of researchFlashes)if(until<=Date.now())researchFlashes.delete(id);
  if(data?.collector?.status!=="unavailable"&&data)researchInitialized=true;
  notifySignals(incoming);
  researchLastUpdate=Date.now();
  $("research-health").textContent=data?.collector?.label||"研究数据暂不可用";
  $("research-health").className=live?"pill":"pill muted";
  $("research-health").title=data?.collector?.last_heartbeat_at?`采集心跳 ${researchTime(data.collector.last_heartbeat_at)}`:"";
  $("research-rule").textContent=data?.rule_note||"独立研究账本，与原AI方向分开记录";
  $("research-history").textContent=data?.history_note||"";$("research-source").textContent=data?.source_note||"Coinbase ETH-USD · 不混用OKX ETH-USDC报价";
  $("research-list").setAttribute("data-stale",String(!live));
  const signature=JSON.stringify([signals,live,data?.families,[...researchFlashes.keys()]]);
  if(signature!==researchSignature){
    researchSignature=signature;
    if(data?.families)$("research-list").replaceChildren(...data.families.map(researchTile));
    else if(signals.length)$("research-list").replaceChildren(...signals.map(researchCard));
    else{const p=document.createElement("p");p.className="subtitle";p.textContent=live?"尚无符合条件的研究信号，继续等待形态形成。":"研究数据暂不可用或采集尚未就绪；不是观望判决。";$("research-list").replaceChildren(p);}
  }
}
function render(s){
  notifyDirections(s);
  renderMacro(s.macro_events);
  renderResearch(s.research_signals);
  const t4=s["4h"]||{}, t15=s["15m"]||{}, c4=t4.context||{}, b4=t4.brooks||{}, d4=b4.deep||{}, b15=t15.brooks||{}, d15=b15.deep||{}, lv=t15.levels||{}, i4=t4.indicators||{}, i15=t15.indicators||{};
  $("health").className="pill"; $("health").innerHTML="<i></i> 实时通信"; $("health").title=`最近快照 ${new Date(s.updated_at).toLocaleTimeString()}`; $("price").textContent=`$${fmt(s.price)}`; $("news").textContent=plainMacroText(s.news);
  renderDirection("4",t4);renderDirection("15",t15);
  $("price-params").textContent=`15m量比 ${fmt(i15["量比"])} · 4h ATR ${fmt(i4.ATR)}%`;
  $("price-updated").textContent=`页面快照 ${macroTime(s.updated_at)} · 非成交指令`;
  $("trend4").textContent=structureText(d4.trend_leg,t4.direction); $("squeeze").textContent=`挤压 ${fmt(c4.squeeze_pct)}% · ATR ${fmt(c4.atr4h_pct)}% · 4小时收线更新`;
  $("trend15").textContent=structureText(d15.trend_leg,t15.direction); $("bar15").textContent=`${(d15.bar_read||[]).join("；")||"暂无K线判断"} · 15分钟收线更新`;
  $("support").textContent=fmt(lv.swing_low); $("resistance").textContent=fmt(lv.swing_high);
  $("support4").textContent=fmt(t4.levels?.support);$("resistance4").textContent=fmt(t4.levels?.resistance);
  $("risk4").textContent=d4.risk||"—"; $("risk15").textContent=d15.risk||"—";
  putModules("modules4",[
    ...evidenceRows(b4),
    ["核心 · Brooks",structureText(d4.trend_leg,t4.direction),[...(d4.bar_read||[]),...(d4.structure||[])].join("；")||"暂无结构结论"],
    ["趋势",`EMA ${i4["EMA排列"]||"—"} · RSI ${fmt(i4.RSI)}`,i4["RSI历史"]?.text||"暂无统计"],
    ["波动率",`挤压 ${fmt(i4["挤压分位"])}% · ATR ${fmt(i4.ATR)}%`,`ADX ${fmt(i4.ADX)} · 只判断波动强弱`],
    ["量能",`量比 ${fmt(i4["量比"])}`,"用于确认突破，不单独决定方向"],
    ["区间 · 威科夫",c4.wyckoff?`${c4.wyckoff.support} — ${c4.wyckoff.resistance}`:"暂无区间","辅助识别吸筹、派发和区间突破"]
  ]);
  putModules("modules15",[
    ...evidenceRows(b15),
    ["核心 · Brooks",structureText(d15.trend_leg,t15.direction),[...(d15.bar_read||[]),...(d15.structure||[])].join("；")||"暂无结构结论"],
    ["动量",`RSI ${fmt(i15.RSI)} · ADX ${fmt(i15.ADX)}`,i15["RSI历史"]?.text||"暂无统计"],
    ["短线趋势",`EMA20 ${fmt(i15.EMA20)} · EMA50 ${fmt(i15.EMA50)}`,"只服务15分钟入场，不代替4小时方向"],
    ["量能与成本",`量比 ${fmt(i15["量比"])} · VWAP ${i15.VWAP?fmt(i15.VWAP[0]):"—"}`,i15.VWAP?`距VWAP ${fmt(i15.VWAP[1])}%`:"暂无VWAP"],
    ["形态确认",(b15.setups||[]).join("；")||"无",`K线重叠 ${Number.isFinite(d15.overlap)?Math.round(d15.overlap*100)+"%":"—"}`]
  ]);
  renderJournal(s.journal);
}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);render(await r.json())}catch(e){$("health").textContent='通信中断';$("health").className='pill muted'}}
let macroClock=null,macroSignature="";
function macroText(tag,text,className=""){
  const element=document.createElement(tag);element.textContent=plainMacroText(text);element.className=className;return element;
}
function macroTime(value){
  const date=new Date(value);return Number.isFinite(date.getTime())?date.toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false}):"时间未知";
}
function macroCard(event){
  const card=macroText("article","","macro-card");card.dataset.priority=event.priority;card.dataset.release=event.release_at;
  card.append(macroText("h3",event.name),macroText("div",`北京 ${macroTime(event.release_at)}`,"macro-time"),macroText("span","","macro-countdown"));
  const metrics=event.metrics||[];
  if(metrics.length){
    const table=document.createElement("table");table.className="macro-table";
    const head=document.createElement("tr");for(const value of ["分项","市场预期","前值"])head.append(macroText("th",value));
    const thead=document.createElement("thead");thead.append(head);table.append(thead);
    const body=document.createElement("tbody");
    for(const q of metrics){const row=document.createElement("tr");for(const value of [q.label,q.forecast??"未获取",q.prior_status==="conflict"?"待核对 ⚠":q.previous??"未获取"])row.append(macroText("td",value));body.append(row);}
    table.append(body);card.append(table);
  }else card.append(macroText("p","市场预期 — · 前值 —\n尚无已获取的事前数字","macro-missing"));
  const state={fresh:"事前采集",frozen:"公布前留档",stale:"旧参考 · 抓取已过期",missing:"预期未获取",conflict:"前值口径冲突 · 禁止直接比较"};
  card.append(macroText("div",`${state[event.expectation_status]||"状态未知"}${event.captured_at?` · ${macroTime(event.captured_at)}`:""}`,"macro-capture"));
  const detail=document.createElement("details");detail.append(macroText("summary","预案与依据"));
  for(const text of [...metrics.map(q=>`${q.label}：${q.comparison}`),...(event.scenarios||[]),"预期相对前值的变化不代表市场已经定价；不是买卖指令。"]){detail.append(macroText("p",text));}
  try{const url=new URL(event.calendar_source);if(url.protocol==="https:"){const a=macroText("a","日历来源");a.href=url.href;a.target="_blank";a.rel="noopener noreferrer";detail.append(a);}}catch(_){/* 无有效来源时不生成链接。 */}
  card.append(detail);return card;
}
function tickMacro(){
  if(!macroClock||!$("macro-list"))return;
  const elapsed=performance.now()-macroClock.started,stale=elapsed>20000;
  $("macro-list").dataset.stale=String(stale);
  if(stale)$("macro-status").textContent="页面通信过期 · 旧预期仅供参考";
  for(const card of $("macro-list").querySelectorAll(".macro-card")){
    const seconds=Math.ceil((Date.parse(card.dataset.release)-macroClock.server-elapsed)/1000);
    let text="公布时间未知";
    if(Number.isFinite(seconds))text=seconds<=0?"已到计划公布时间 · 等待核实实际值":`还有 ${Math.floor(seconds/86400)?`${Math.floor(seconds/86400)}天 `:""}${Math.floor(seconds%86400/3600)}小时 ${Math.floor(seconds%3600/60)}分 ${seconds%60}秒`;
    card.querySelector(".macro-countdown").textContent=stale?"通信过期 · 请重连核对时间":text;
  }
}
function renderMacro(data){
  if(!$("macro-list"))return;
  if(!data){macroClock=null;$("macro-status").textContent="事件数据暂不可用";$("macro-list").dataset.stale="true";return;}
  const server=Date.parse(data.updated_at);
  if(!Number.isFinite(server)){macroClock=null;$("macro-status").textContent="事件时间无效";$("macro-list").dataset.stale="true";return;}
  macroClock={server,started:performance.now()};
  const all=(data.events||[]).filter(e=>Number.isFinite(Date.parse(e.release_at))).sort((a,b)=>Date.parse(a.release_at)-Date.parse(b.release_at));
  const near=all.filter(e=>Date.parse(e.release_at)>=server-7200000&&Date.parse(e.release_at)<=server+86400000);
  const events=near.filter(e=>Date.parse(e.release_at)===Date.parse(near[0]?.release_at)),hasQuotes=events.some(e=>e.metrics?.length);
  const missing=events.some(e=>!e.metrics?.length||e.metrics.some(q=>q.forecast==null));
  const state=data.collector_status==="ok"?(hasQuotes?(missing?"部分预期已采集":"事前预期已采集"):"部分事件预期暂缺"):data.collector_status==="error"?"预期采集失败 · 保留旧参考":"预期尚未采集";
  $("macro-status").textContent=data.calendar_stale?"日历需复核 · 预期仅供参考":state;
  if(!events.length&&data.collector_status==="ok"&&!data.calendar_stale)$("macro-status").textContent="暂无临近事件";
  if(events.some(e=>e.expectation_status==="conflict"))$("macro-status").textContent+=" · 部分前值待核对";
  const next=all.find(e=>Date.parse(e.release_at)>server);
  $("macro-note").textContent=events.length?"北京时间 · 最近一组 · 同时公布分项合并；其余事件临近后展开":next?`下一组：${next.name} · ${macroTime(next.release_at)}，24小时内展开`:"未来24小时暂无已核验事件，不能据此排除突发消息";
  $("macro-source").textContent=`${data.source||"预期来源待确认"} · ${data.note||"不是实时实际值通道"}`;
  const signature=JSON.stringify(events.map(({seconds_to_release,...rest})=>rest));
  if(signature!==macroSignature){
    macroSignature=signature;
    $("macro-list").replaceChildren(...events.map(macroCard));
  }
  tickMacro();
}
setInterval(tickMacro,1000);
let socket;
function connect(){
  const protocol=location.protocol==="https:"?"wss":"ws";
  socket=new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage=(event)=>{try{render(JSON.parse(event.data))}catch(_){}};
  socket.onerror=()=>socket.close();
  socket.onclose=()=>{$("health").textContent="通信重连中";$("health").className="pill muted";setTimeout(connect,2000)};
}
$("research-sound")?.addEventListener("click",toggleResearchSound);
$("research-test")?.addEventListener("click",testResearchSound);
$("signal-alarm-stop")?.addEventListener("click",()=>stopSignalAlarm());
setInterval(tickSignalAlarm,1000);
setInterval(()=>{if(researchLastUpdate&&Date.now()-researchLastUpdate>20000){$("research-health").textContent="页面通信过期 · 请勿追旧信号";$("research-health").className="pill muted";$("research-list").style.opacity=".55";$("research-list").setAttribute("data-stale","true");}else if($("research-list"))$("research-list").style.opacity="1";},2000);
refresh(); connect(); setInterval(()=>{if(!socket||socket.readyState!==WebSocket.OPEN)refresh()},30000);
