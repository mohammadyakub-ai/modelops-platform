"""Self-contained live-demo page served by the modelops API at /demo.

Purpose: one URL that IS the live demonstration — model version, measured
latency, prediction distribution, feature drift (PSI) and the retraining alert,
all pulled live from /metrics + /model/info with zero external assets (works
behind any host, including the sealed Render free tier).
"""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>modelops — live serving demo</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--bad:#f85149;--warn:#d29922;--acc:#58a6ff}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;padding:24px}
  h1{font-size:20px;margin-bottom:4px}
  .sub{color:var(--dim);font-size:12px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
  .card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);margin-bottom:10px}
  .k{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #21262d}
  .k:last-child{border-bottom:none}
  .v{font-variant-numeric:tabular-nums;font-weight:600}
  .pill{border-radius:999px;padding:2px 10px;font-size:12px;font-weight:700}
  .pill.ok{background:#122b1e;color:var(--ok)}
  .pill.bad{background:#2d1215;color:var(--bad)}
  .pill.warn{background:#2d2512;color:var(--warn)}
  .bar{height:8px;background:#21262d;border-radius:4px;overflow:hidden;margin:4px 0 8px}
  .bar>i{display:block;height:100%;background:var(--acc);border-radius:4px}
  canvas{width:100%;height:120px;background:#010409;border:1px solid var(--line);border-radius:8px}
  .psi{display:grid;grid-template-columns:auto 1fr 54px;gap:8px;align-items:center;font-size:12px;margin:6px 0}
  .psi .nm{color:var(--dim);min-width:94px}
  .psi .sp{height:7px;background:#21262d;border-radius:4px;overflow:hidden}
  .psi .sp>i{display:block;height:100%;background:var(--ok)}
  .psi.flagged .sp>i{background:var(--bad)} .psi.flagged .nm{color:var(--bad)}
  .psi .val{text-align:right;font-variant-numeric:tabular-nums}
  .foot{color:var(--dim);font-size:11px;margin-top:24px;text-align:center}
  a{color:var(--acc);text-decoration:none}
</style>
</head>
<body>
<h1>modelops — live serving demonstration</h1>
<div class="sub">one URL, the whole lifecycle: production model, measured latency, predictions, drift</div>
<div class="grid">
  <div class="card">
    <h2>Production model</h2>
    <div id="model" class="k"><span class="v">loading…</span></div>
    <div id="ready" style="margin-top:10px"></div>
  </div>
  <div class="card">
    <h2>Latency (measured on this server)</h2>
    <div id="lat" class="k"><span class="v">…</span></div>
  </div>
  <div class="card">
    <h2>Traffic</h2>
    <div id="traffic" class="k"><span class="v">…</span></div>
  </div>
</div>
<div class="grid" style="margin-top:12px">
  <div class="card">
    <h2>Positive rate over time (live)</h2>
    <canvas id="rate"></canvas>
  </div>
  <div class="card">
    <h2>Feature drift — PSI (threshold 0.20)</h2>
    <div id="psi">no drift report yet</div>
  </div>
</div>
<div class="foot">
  ModelOps Platform · <a href="/model/info">/model/info</a> ·
  <a href="/metrics">/metrics</a> · <a href="/docs">OpenAPI</a> ·
  drift &rarr; retraining alert levels shown live above
</div>
<script>
const $=id=>document.getElementById(id);
const state={rate:{x:0,y:0},history:[],totals:{}};
function fmt(n,d=2){if(n==null||isNaN(n))return"—";return(+n).toFixed(d)}
function pill(txt,cls){return `<span class="pill ${cls}">${txt}</span>`}
function kv(k,v){return `<div class="k"><span>${k}</span><span class="v">${v}</span></div>`}

async function tick(){
  const [info,metrics]=await Promise.all([fetch('/model/info').then(r=>r.json()),
    fetch('/metrics').then(r=>r.text()).catch(()=>"")]);
  // model card
  const name=info.name||info.model||"—", ver=info.version;
  $('model').innerHTML=kv("model",`<code>${name}</code>`)+kv("version",`v${ver}`)+kv("stage",info.stage||"Production")+
    kv("metrics","auc "+fmt((info.metrics||{}).roc_auc,4)+" · acc "+fmt((info.metrics||{}).accuracy,4));
  // readiness pill
  const ready=await fetch('/ready').then(r=>r.status===200).catch(()=>false);
  $('ready').innerHTML=ready?pill("READY","ok"):pill("NOT READY","bad");
  // latency from live server measurements
  const lat=info.served_latency_ms;
  $('lat').innerHTML=(lat?kv("requests","n="+lat.n_requests)+kv("p50",fmt(lat.p50,2)+" ms")+
    kv("p95",fmt(lat.p95,2)+" ms")+kv("p99",fmt(lat.p99,2)+" ms"):"no traffic yet — POST /predict"); 
  // traffic counters from /metrics
  let tot=0,pos=0,errs=0,rate=0;
  const drift={},psi={};
  for(const line of metrics.split("\n")){
    if(line.includes('modelops_predictions_total{')){
      const c=parseFloat(line.split(" ").pop()); tot+=c;
      if(/label="1"/.test(line))pos=c;
    }
    if(line.startsWith('modelops_prediction_errors_total'))errs+=parseFloat(line.split(" ").pop());
    if(line.startsWith('modelops_positive_rate'))rate=parseFloat(line.split(" ").pop());
    if(line.startsWith('modelops_feature_psi{')){const m=line.match(/feature="([^"]+)"/);psi[m[1]]=parseFloat(line.split(" ").pop())}
    if(line.startsWith('modelops_drift_alert'))drift.alert=parseInt(line.split(" ").pop());
  }
  $('traffic').innerHTML=kv("predictions",fmt(tot,0))+kv("errors",fmt(errs,0))+kv("positive rate",fmt(rate*100,1)+"%");
  // positive-rate history
  state.history.push(rate); if(state.history.length>80)state.history.shift();
  drawRate(state.history);
  // drift
  if(Object.keys(psi).length){
    let rows="";
    for(const [f,v] of Object.entries(psi)){const flag=v>0.2;
      rows+=`<div class="psi ${flag?'flagged':''}"><span class="nm">${f}</span><span class="sp"><i style="width:${Math.min(100,v/0.6*100)}%"></i></span><span class="val">${fmt(v,3)}</span></div>`}
    $('psi').innerHTML=rows+`<div style="margin-top:8px">${drift.alert?pill("RETRAIN ALERT","bad"):pill("NO DRIFT","ok")}</div>`;
  }
}
function drawRate(h){
  const c=$('rate');if(!c)return;const ctx=c.getContext("2d"),w=c.width=Math.floor(c.clientWidth),ht=c.height=120;
  ctx.clearRect(0,0,w,ht);ctx.strokeStyle="#30363d";ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,ht/2);ctx.lineTo(w,ht/2);ctx.stroke();
  ctx.strokeStyle="#58a6ff";ctx.lineWidth=2;ctx.beginPath();
  h.forEach((y,i)=>{const x=i/(h.length-1||1)*w,py=ht-(y*ht);i?ctx.lineTo(x,py):ctx.moveTo(x,py)});ctx.stroke();
  ctx.fillStyle="#58a6ff";ctx.font="11px monospace";
  const last=h[h.length-1]||0;ctx.fillText("positive rate source: modelops_positive_rate",4,12);
}
tick();setInterval(tick,3000);
</script>
</body>
</html>
"""


def demo_page() -> str:
    return DEMO_HTML