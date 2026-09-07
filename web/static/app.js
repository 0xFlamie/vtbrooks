const $ = (id) => document.getElementById(id);
const cnDir = (v) => ({bull_leg:"多头腿",bear_leg:"空头腿",range:"震荡/无连续方向"}[v] || v || "—");
const fmt = (v) => typeof v === "number" ? v.toFixed(2) : (v ?? "—");
const dirText = (v) => v === "LONG" ? "做多" : v === "SHORT" ? "做空" : "观望";
const dirClass = (v) => v === "LONG" ? "long" : v === "SHORT" ? "short" : "flat";
function putData(id, rows){ $(id).innerHTML = rows.map(([k,v]) => `<div><span>${k}</span><b>${v}</b></div>`).join(""); }
function render(s){
  const t4=s["4h"]||{}, t15=s["15m"]||{}, c4=t4.context||{}, b4=t4.brooks||{}, d4=b4.deep||{}, b15=t15.brooks||{}, d15=b15.deep||{}, lv=t15.levels||{}, i4=t4.indicators||{}, i15=t15.indicators||{};
  $("updated").textContent=`更新 ${new Date(s.updated_at).toLocaleTimeString()}`; $("price").textContent=`$${fmt(s.price)}`; $("news").textContent=s.news;
  $("direction4").textContent=dirText(t4.direction); $("direction4").className=dirClass(t4.direction); $("direction4").closest(".direction").className=`direction panel ${dirClass(t4.direction)}`; $("confidence4").textContent=t4.confidence == null ? "暂无判决" : `置信 ${t4.confidence}%`; $("reason4").textContent=(t4.latest_reasons||[]).join("；")||"原因随下一次4h判决更新";
  $("direction15").textContent=dirText(t15.direction); $("direction15").className=dirClass(t15.direction); $("direction15").closest(".direction").className=`direction panel ${dirClass(t15.direction)}`; $("confidence15").textContent=t15.confidence == null ? "暂无判决" : `置信 ${t15.confidence}%`; $("reason15").textContent=(t15.latest_reasons||[]).join("；")||"原因随下一次15m判决更新";
  $("trend4").textContent=cnDir(d4.trend_leg); $("squeeze").textContent=`挤压 ${fmt(c4.squeeze_pct)}% · ATR ${fmt(c4.atr4h_pct)}%`;
  $("trend15").textContent=cnDir(d15.trend_leg); $("bar15").textContent=(d15.bar_read||[]).join("；")||"暂无K线判断";
  $("support").textContent=fmt(lv.swing_low); $("resistance").textContent=fmt(lv.swing_high);
  $("structure4").textContent=[...(d4.bar_read||[]),...(d4.structure||[])].join("；")||"暂无结构结论"; $("risk4").textContent=d4.risk||"—";
  $("structure15").textContent=[...(d15.bar_read||[]),...(d15.structure||[])].join("；")||"暂无结构结论"; $("risk15").textContent=d15.risk||"—";
  putData("data4",[["RSI",`${fmt(i4.RSI)} · ${i4["RSI历史"].text}`],["EMA排列",i4["EMA排列"]||"—"],["挤压分位",`${fmt(i4["挤压分位"])}%`],["ADX",fmt(i4.ADX)],["ATR",`${fmt(i4.ATR)}%`],["量比",fmt(i4["量比"])],["Brooks",[...(d4.bar_read||[]),...(d4.structure||[])].join("；")||"—"],["威科夫TR",c4.wyckoff?`${c4.wyckoff.support} — ${c4.wyckoff.resistance}`:"—"]]);
  putData("data15",[["RSI",`${fmt(i15.RSI)} · ${i15["RSI历史"].text}`],["EMA20 / EMA50",`${fmt(i15.EMA20)} / ${fmt(i15.EMA50)}`],["ADX",fmt(i15.ADX)],["量比",fmt(i15["量比"])],["VWAP",i15.VWAP?`${fmt(i15.VWAP[0])} (${fmt(i15.VWAP[1])}%)`:"—"],["Brooks",[...(d15.bar_read||[]),...(d15.structure||[])].join("；")||"—"],["形态",(b15.setups||[]).join("；")||"无"],["重叠度",`${Math.round((d15.overlap||0)*100)}%`]]);
  $("health").textContent=`15m ${s.health.klines_15m}根 · 4h ${s.health.context_4h?"正常":"异常"}`;
  $("journal").innerHTML=(s.journal||[]).slice().reverse().map(x=>`<div>${x.time||""} · 4h ${x.dir4h||"观望"} / 15m ${x.dir15m||"观望"} · ${x.reasons||[]}</div>`).join("")||"暂无判决记录";
}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});render(await r.json())}catch(e){$("updated").textContent='观察服务不可用'}}
refresh(); setInterval(refresh,30000);
