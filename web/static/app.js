const $ = (id) => document.getElementById(id);
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
const researchSeen=new Set();
let researchInitialized=false, researchAudio=null, researchSoundEnabled=false, researchLastUpdate=0, researchSignature="";
const researchTime=v=>new Date(v).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false});
function researchTone(preview=false){
  if(!researchAudio||researchAudio.state!=="running")return false;
  const start=researchAudio.currentTime;
  for(const [delay,hz] of (preview?[[0,660]]:[[0,660],[.2,880]])){
    const oscillator=researchAudio.createOscillator(), gain=researchAudio.createGain();
    oscillator.type="sine";oscillator.frequency.value=hz;gain.gain.setValueAtTime(0,start+delay);
    gain.gain.linearRampToValueAtTime(.09,start+delay+.015);gain.gain.exponentialRampToValueAtTime(.001,start+delay+.16);
    oscillator.connect(gain);gain.connect(researchAudio.destination);oscillator.start(start+delay);oscillator.stop(start+delay+.18);
  }
  return true;
}
async function toggleResearchSound(){
  const button=$("research-sound");
  if(researchSoundEnabled){researchSoundEnabled=false;button.textContent="开启研究信号声音";button.className="pill muted";button.setAttribute("aria-pressed","false");return;}
  try{
    const Audio=window.AudioContext||window.webkitAudioContext;
    if(!Audio)throw new Error("unsupported");
    researchAudio=researchAudio||new Audio();await researchAudio.resume();
    if(researchAudio.state!=="running")throw new Error("suspended");
    researchSoundEnabled=true;button.textContent="研究信号声音已开启";button.className="pill";button.setAttribute("aria-pressed","true");researchTone(true);
    $("research-sound-note").textContent="已试听提示音；只响新信号，不重播历史。刷新后需重新开启；后台休眠或静音可能漏提醒。";
  }catch(_){researchSoundEnabled=false;button.setAttribute("aria-pressed","false");$("research-sound-note").textContent="声音未能开启，请检查浏览器声音权限；仍可查看卡片。";}
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
function renderResearch(data){
  if(!$("research-list"))return;
  const live=data?.collector?.status==="live", signals=data?.signals||[], serverAt=Date.parse(data?.updated_at);
  let incoming=false;
  for(const signal of signals){
    if(researchInitialized&&!researchSeen.has(signal.id)&&live&&signal.status==="new"&&serverAt<Date.parse(signal.entry_not_before)&&
       Date.now()<Date.parse(signal.entry_not_before)&&Date.now()-serverAt>=-5000&&Date.now()-serverAt<=20000)incoming=true;
    researchSeen.add(signal.id);
  }
  while(researchSeen.size>1000)researchSeen.delete(researchSeen.values().next().value);
  if(data?.collector?.status!=="unavailable"&&data)researchInitialized=true;
  if(incoming&&researchSoundEnabled&&!researchTone()){
    researchSoundEnabled=false;$("research-sound").textContent="重新开启研究信号声音";$("research-sound").setAttribute("aria-pressed","false");
    $("research-sound-note").textContent="有新研究信号，但浏览器暂停了声音；请重新点击开启。";
  }
  researchLastUpdate=Date.now();
  $("research-health").textContent=data?.collector?.label||"研究数据暂不可用";
  $("research-health").className=live?"pill":"pill muted";
  $("research-health").title=data?.collector?.last_heartbeat_at?`采集心跳 ${researchTime(data.collector.last_heartbeat_at)}`:"";
  $("research-rule").textContent=data?.rule_note||"独立研究账本，与原AI方向分开记录";
  $("research-history").textContent=data?.history_note||"";$("research-source").textContent=data?.source_note||"Coinbase ETH-USD · 不混用OKX ETH-USDC报价";
  const signature=JSON.stringify([signals,live]);
  if(signature!==researchSignature){
    researchSignature=signature;
    if(signals.length)$("research-list").replaceChildren(...signals.map(researchCard));
    else{const p=document.createElement("p");p.className="subtitle";p.textContent=live?"尚无符合条件的研究信号，继续等待形态形成。":"研究数据暂不可用或采集尚未就绪；不是观望判决。";$("research-list").replaceChildren(p);}
  }
}
function render(s){
  renderResearch(s.research_signals);
  const t4=s["4h"]||{}, t15=s["15m"]||{}, c4=t4.context||{}, b4=t4.brooks||{}, d4=b4.deep||{}, b15=t15.brooks||{}, d15=b15.deep||{}, lv=t15.levels||{}, i4=t4.indicators||{}, i15=t15.indicators||{};
  $("health").className="pill"; $("health").innerHTML="<i></i> 实时通信"; $("health").title=`最近快照 ${new Date(s.updated_at).toLocaleTimeString()}`; $("price").textContent=`$${fmt(s.price)}`; $("news").textContent=s.news;
  $("direction4").textContent=dirText(t4.direction); $("direction4").className=dirClass(t4.direction); $("direction4").closest(".direction").className=`direction panel ${dirClass(t4.direction)}`; $("confidence4").textContent=t4.confidence == null ? "暂无判决" : `置信 ${t4.confidence}%`; $("reason4").textContent=t4.summary||((t4.latest_reasons||[]).join("；"))||"原因随下一次4h判决更新";
  $("direction15").textContent=dirText(t15.direction); $("direction15").className=dirClass(t15.direction); $("direction15").closest(".direction").className=`direction panel ${dirClass(t15.direction)}`; $("confidence15").textContent=t15.confidence == null ? "暂无判决" : `置信 ${t15.confidence}%`; $("reason15").textContent=t15.summary||((t15.latest_reasons||[]).join("；"))||"原因随下一次15m判决更新";
  $("trend4").textContent=structureText(d4.trend_leg,t4.direction); $("squeeze").textContent=`挤压 ${fmt(c4.squeeze_pct)}% · ATR ${fmt(c4.atr4h_pct)}% · 4小时收线更新`;
  $("trend15").textContent=structureText(d15.trend_leg,t15.direction); $("bar15").textContent=`${(d15.bar_read||[]).join("；")||"暂无K线判断"} · 15分钟收线更新`;
  $("support").textContent=fmt(lv.swing_low); $("resistance").textContent=fmt(lv.swing_high);
  $("risk4").textContent=d4.risk||"—"; $("risk15").textContent=d15.risk||"—";
  const planText=(p,d,note)=>p?`进场参考 $${fmt(p.entry)} · 止损 $${fmt(p.sl)} · 一目标 $${fmt(p.tp1)} · 二目标 $${fmt(p.tp2)}<small>${p.rule}</small>`:`${note||`${dirText(d)}：暂不进场，不给虚假止盈止损点位`}`;
  $("plan4").innerHTML=`<b>Brooks进出场计划</b>${planText(t4.plan,t4.direction,t4.plan_note)}`; $("plan15").innerHTML=`<b>Brooks进出场计划</b>${planText(t15.plan,t15.direction,t15.plan_note)}`;
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
    ["形态确认",(b15.setups||[]).join("；")||"无",`K线重叠 ${Math.round((d15.overlap||0)*100)}%`]
  ]);
  $("journal").innerHTML=(s.journal||[]).slice().reverse().map(x=>`<div>${x.time||""} · 4h ${x.dir4h||"观望"} / 15m ${x.dir15m||"观望"} · ${x.reasons||[]}</div>`).join("")||"暂无判决记录";
}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});if(!r.ok)throw new Error(`HTTP ${r.status}`);render(await r.json())}catch(e){$("health").textContent='通信中断';$("health").className='pill muted'}}
let socket;
function connect(){
  const protocol=location.protocol==="https:"?"wss":"ws";
  socket=new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage=(event)=>{try{render(JSON.parse(event.data))}catch(_){}};
  socket.onerror=()=>socket.close();
  socket.onclose=()=>{$("health").textContent="通信重连中";$("health").className="pill muted";setTimeout(connect,2000)};
}
$("research-sound")?.addEventListener("click",toggleResearchSound);
setInterval(()=>{if(researchLastUpdate&&Date.now()-researchLastUpdate>20000){$("research-health").textContent="页面通信过期 · 请勿追旧信号";$("research-health").className="pill muted";$("research-list").style.opacity=".55";}else if($("research-list"))$("research-list").style.opacity="1";},2000);
refresh(); connect(); setInterval(()=>{if(!socket||socket.readyState!==WebSocket.OPEN)refresh()},30000);
