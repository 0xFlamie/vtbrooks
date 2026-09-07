const $ = (id) => document.getElementById(id);
const cnDir = (v) => ({bull_leg:"多头腿",bear_leg:"空头腿",range:"震荡/无连续方向"}[v] || v || "—");
const fmt = (v) => typeof v === "number" ? v.toFixed(2) : (v ?? "—");
function putData(id, rows){ $(id).innerHTML = rows.map(([k,v]) => `<div><span>${k}</span><b>${v}</b></div>`).join(""); }
function render(s){
  const c4=s["4h"].context||{}, b4=s["4h"].brooks||{}, d4=b4.deep||{}, b15=s["15m"].brooks||{}, d15=b15.deep||{}, lv=s["15m"].levels||{};
  $("updated").textContent=`更新 ${new Date(s.updated_at).toLocaleTimeString()}`; $("price").textContent=`$${fmt(s.price)}`; $("news").textContent=s.news;
  $("trend4").textContent=cnDir(d4.trend_leg); $("squeeze").textContent=`挤压 ${fmt(c4.squeeze_pct)}% · ATR ${fmt(c4.atr4h_pct)}%`;
  $("trend15").textContent=cnDir(d15.trend_leg); $("bar15").textContent=(d15.bar_read||[]).join("；")||"暂无K线判断";
  $("support").textContent=fmt(lv.swing_low); $("resistance").textContent=fmt(lv.swing_high);
  $("structure4").textContent=[...(d4.bar_read||[]),...(d4.structure||[])].join("；")||"暂无结构结论"; $("risk4").textContent=d4.risk||"—";
  $("structure15").textContent=[...(d15.bar_read||[]),...(d15.structure||[])].join("；")||"暂无结构结论"; $("risk15").textContent=d15.risk||"—";
  putData("data4",[["Always In",({1:"多","-1":"空"}[b4.always_in]||"—")],["位置",d4.location||"—"],["重叠度",`${Math.round((d4.overlap||0)*100)}%`],["威科夫TR",c4.wyckoff?`${c4.wyckoff.support} — ${c4.wyckoff.resistance}`:"—"]]);
  putData("data15",[["形态",(b15.setups||[]).join("；")||"无"],["位置",d15.location||"—"],["重叠度",`${Math.round((d15.overlap||0)*100)}%`],["RSI",fmt(lv.rsi14)]]);
  $("health").textContent=`15m ${s.health.klines_15m}根 · 4h ${s.health.context_4h?"正常":"异常"}`;
  $("journal").innerHTML=(s.journal||[]).slice().reverse().map(x=>`<div>${x.time||""} · 4h ${x.dir4h||"观望"} / 15m ${x.dir15m||"观望"} · ${x.reasons||[]}</div>`).join("")||"暂无判决记录";
}
async function refresh(){try{const r=await fetch('/api/snapshot',{cache:'no-store'});render(await r.json())}catch(e){$("updated").textContent='观察服务不可用'}}
refresh(); setInterval(refresh,30000);
