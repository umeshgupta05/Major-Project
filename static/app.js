/* ===== MSTCT Pipeline — Graphical Operation Visualizations ===== */

const charts={};
function fmt(v,d=4){return(v==null||isNaN(Number(v)))?'—':Number(v).toFixed(d);}
function $(id){return document.getElementById(id);}
function destroyChart(k){if(charts[k]){charts[k].destroy();delete charts[k];}}
function delay(ms){return new Promise(r=>setTimeout(r,ms));}
const C={accent:'#10b981',accentFill:'rgba(16,185,129,0.15)',amber:'#f59e0b',amberFill:'rgba(245,158,11,0.15)',red:'#ef4444',blue:'#3b82f6',grid:'rgba(255,255,255,0.05)',tick:'#7a9aab'};
function chartOpts(){return{responsive:true,maintainAspectRatio:true,plugins:{legend:{labels:{color:C.tick,font:{family:"'Inter',sans-serif",size:11}}}},scales:{x:{ticks:{color:C.tick,font:{size:10}},grid:{color:C.grid}},y:{ticks:{color:C.tick,font:{size:10}},grid:{color:C.grid}}}};}

/* ===== PARTICLES ===== */
function initParticles(){
  const cv=$('particleCanvas'),ctx=cv.getContext('2d');let w,h;const pts=[];
  function resize(){w=cv.width=innerWidth;h=cv.height=innerHeight;}
  resize();addEventListener('resize',resize);
  for(let i=0;i<35;i++)pts.push({x:Math.random()*w,y:Math.random()*h,r:Math.random()*1.5+0.4,dx:(Math.random()-0.5)*0.25,dy:(Math.random()-0.5)*0.25,o:Math.random()*0.25+0.08});
  (function draw(){ctx.clearRect(0,0,w,h);pts.forEach(p=>{p.x+=p.dx;p.y+=p.dy;if(p.x<0)p.x=w;if(p.x>w)p.x=0;if(p.y<0)p.y=h;if(p.y>h)p.y=0;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(16,185,129,${p.o})`;ctx.fill();});requestAnimationFrame(draw);})();
}

/* ===== STAGE MANAGEMENT ===== */
let activeStage=0;
function setTracker(s){document.querySelectorAll('.step').forEach((el,i)=>{el.classList.remove('active','completed');if(i<s)el.classList.add('completed');else if(i===s)el.classList.add('active');});document.querySelectorAll('.step__line').forEach((el,i)=>{el.classList.toggle('completed',i<s);});$('trackerProgress').style.width=`${(s/5)*100}%`;}
async function goToStage(s){if(s===activeStage)return;$(`stage-${activeStage}`).classList.remove('active');$(`stage-${activeStage}`).classList.add('exit-left');await delay(150);$(`stage-${s}`).classList.remove('exit-left');$(`stage-${s}`).classList.add('active');activeStage=s;setTracker(s);$(`stage-${s}`).scrollTop=0;}
function showOp(id){const el=$(id);if(el){el.style.display='';el.classList.add('visible');}}

/* ===== MODEL INFO ===== */
async function loadOverview(){try{const r=await fetch('/api/overview');const d=await r.json();renderModelInfo(d);}catch{$('artifactStatus').textContent='Could not load artifacts.';}}
function renderModelInfo(data){
  $('artifactStatus').textContent=data.model_ready?'Artifacts loaded — model + scaler + feature config ✓':`Status: ${data.artifact_status}`;
  const fg=data.feature_groups||{},m=data.metrics||{},d=data.dataset||{};
  $('modelArchContent').innerHTML=`<div class="arch-grid"><div class="arch-item"><span class="label">Architecture</span><span class="value">MSTCT</span></div><div class="arch-item"><span class="label">Input Dim</span><span class="value">${data.input_dim||'—'}</span></div><div class="arch-item"><span class="label">Seq Length</span><span class="value">${data.sequence_length||48}</span></div><div class="arch-item"><span class="label">Stride</span><span class="value">${data.stride||12}</span></div><div class="arch-item"><span class="label">Threshold</span><span class="value">${fmt(data.threshold,3)}</span></div><div class="arch-item"><span class="label">Patients</span><span class="value">${(d.patients||0).toLocaleString()}</span></div></div>`;
  $('metricsBar').innerHTML=`<span class="text-accent">●</span> AUROC <strong>${fmt(m.auroc,3)}</strong> · F1 <strong>${fmt(m.f1,3)}</strong> · Recall <strong>${fmt(m.recall,3)}</strong><span style="display:block;margin-top:4px;font-size:11px;color:var(--text-muted);">Features: ${fg.temporal||0} temporal · ${fg.vitals||0} vitals · ${fg.labs||0} labs · ${fg.static||0} static · ${fg.rolling||0} rolling</span>`;
}

/* ===== UPLOAD ===== */
function initUpload(){
  const z=$('uploadZone'),fi=$('csvFile'),form=$('predictForm'),info=$('selectedFileInfo');
  ['dragenter','dragover'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.add('dragover');}));
  ['dragleave','drop'].forEach(e=>z.addEventListener(e,ev=>{ev.preventDefault();z.classList.remove('dragover');}));
  z.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){fi.files=ev.dataTransfer.files;upd();}});
  fi.addEventListener('change',upd);
  function upd(){if(fi.files&&fi.files.length){const f=fi.files[0];info.textContent=`Selected: ${f.name} (${Math.round(f.size/1024)} KB)`;info.style.color='var(--accent)';}}
  form.addEventListener('submit',handlePredict);
  document.querySelectorAll('.step').forEach(s=>{s.addEventListener('click',()=>{const i=parseInt(s.dataset.step);if(i<=activeStage)goToStage(i);});});
}

async function handlePredict(e){
  e.preventDefault();const fi=$('csvFile'),pid=$('patientId').value.trim();
  if(!fi.files.length){alert('Please select a CSV file.');return;}
  const fd=new FormData();fd.append('file',fi.files[0]);fd.append('csvFile',fi.files[0]);
  if(pid)fd.append('patient_id',pid);
  const btn=$('submitBtn'),bt=$('btnText'),sp=$('btnSpinner'),st=$('uploadStatus');
  btn.disabled=true;bt.textContent='Processing…';sp.classList.remove('hidden');
  st.textContent='Uploading and running pipeline…';st.style.color='var(--text-secondary)';
  try{
    const res=await fetch('/api/predict',{method:'POST',headers:{Accept:'application/json'},body:fd});
    const text=await res.text();let payload;
    try{payload=JSON.parse(text);}catch{throw new Error(`Server error: ${text.slice(0,200)}`);}
    if(!res.ok||!payload.ok)throw new Error(payload.error||'Prediction failed');
    if(payload.upload_ack){const a=payload.upload_ack;st.textContent=`✓ ${a.filename} — ${a.rows_received} rows, ${a.columns_received} cols${a.patient_filter?' · Patient: '+a.patient_filter:''}`;st.style.color='var(--accent)';}
    initPipeline(payload);
  }catch(err){st.textContent=`✗ ${err.message||String(err)}`;st.style.color='var(--red)';}
  finally{btn.disabled=false;bt.textContent='Run Pipeline';sp.classList.add('hidden');}
}

/* ===== PIPELINE NAVIGATION ===== */
let pipelinePayload=null;
let maxReachedStage=0;
const renderedStages=new Set();
const stageNames=['Upload','Preprocess','Features','TCN','Transformer','Output'];

function initPipeline(P){
  pipelinePayload=P;
  maxReachedStage=1;
  renderedStages.clear();
  renderedStages.add(0);
  $('stageNav').classList.add('visible');
  navigateToStage(1);
}

async function navigateToStage(s){
  if(s<0||s>5)return;
  await goToStage(s);
  updateNavButtons();
  if(!renderedStages.has(s)){
    renderedStages.add(s);
    if(s>maxReachedStage)maxReachedStage=s;
    await renderStage(s);
  }
}

async function renderStage(s){
  const P=pipelinePayload;if(!P)return;
  switch(s){
    case 1:await renderPreprocess(P);break;
    case 2:await renderFeatures(P);break;
    case 3:await renderTCN(P);break;
    case 4:await renderTransformer(P);break;
    case 5:await renderOutput(P);break;
  }
}

function updateNavButtons(){
  const prev=$('prevBtn'),next=$('nextBtn'),label=$('navLabel');
  prev.disabled=(activeStage<=0);
  next.disabled=(activeStage>=maxReachedStage&&activeStage>=5);
  // If we haven't reached the next stage yet, the Next button should say "Next Stage"
  if(activeStage>=maxReachedStage&&activeStage<5){
    next.disabled=false;
    next.textContent='Next Stage →';
  }else if(activeStage>=5){
    next.disabled=true;
    next.textContent='Complete ✓';
  }else{
    next.textContent='Next →';
  }
  prev.textContent=activeStage<=0?'← Previous':`← ${stageNames[activeStage-1]}`;
  if(activeStage<5&&!next.disabled)next.textContent=`${stageNames[activeStage+1]} →`;
  label.textContent=`Stage ${activeStage+1} of 6 — ${stageNames[activeStage]}`;
}

function initNavButtons(){
  $('prevBtn').addEventListener('click',()=>navigateToStage(activeStage-1));
  $('nextBtn').addEventListener('click',()=>navigateToStage(activeStage+1));
}

/* ================================================================
   STAGE 1: PREPROCESSING — Animated data grid + missing charts
   ================================================================ */
async function renderPreprocess(P){
  const viz=P.preprocessing_viz||{};
  const trace=P.preprocessing_trace||[];
  const s1=trace.find(t=>t.step==='Input validation')||{};
  const rawSample=viz.raw_sample||[];
  const cleanSample=viz.clean_sample||[];

  // Build animated imputation grid from raw sample
  const rawCols=rawSample.length?Object.keys(rawSample[0]).slice(0,8):[];
  const rows=Math.min(rawSample.length,6);

  let gridHtml=`<div class="glass-card"><h3>Data Imputation</h3><p class="muted" style="margin-bottom:12px;">Watch null values (red) get filled with computed values (green)</p><div class="impute-grid">`;
  // Header
  gridHtml+=`<div class="impute-row">${rawCols.map(c=>`<div class="impute-header">${c.length>6?c.slice(0,6)+'…':c}</div>`).join('')}</div>`;
  // Rows
  for(let r=0;r<rows;r++){
    gridHtml+=`<div class="impute-row">`;
    for(let ci=0;ci<rawCols.length;ci++){
      const c=rawCols[ci];
      const rawVal=rawSample[r]?rawSample[r][c]:null;
      const isNull=rawVal==null||rawVal==='';
      gridHtml+=`<div class="impute-cell ${isNull?'missing':'original'}" data-row="${r}" data-col="${ci}" data-raw="${isNull?'null':typeof rawVal==='number'?fmt(rawVal,1):rawVal}">${isNull?'NaN':typeof rawVal==='number'?fmt(rawVal,1):rawVal}</div>`;
    }
    gridHtml+=`</div>`;
  }
  gridHtml+=`</div><div class="impute-counter">Missing cells: <span class="n text-amber" id="missingCounter">${s1.missing_before||0}</span></div></div>`;
  $('prepOp1').innerHTML=gridHtml;
  showOp('prepOp1');

  // Animate cells: after 1.5s, start filling nulls one by one
  await delay(1800);
  const missingCells=document.querySelectorAll('.impute-cell.missing');
  let filled=0;
  for(const cell of missingCells){
    await delay(200);
    cell.classList.remove('missing');
    cell.classList.add('filled');
    cell.textContent='✓';
    filled++;
    const counter=$('missingCounter');
    if(counter){
      const remaining=Math.max(0,(s1.missing_before||missingCells.length)-filled);
      counter.textContent=remaining;
      if(remaining===0){counter.classList.remove('text-amber');counter.classList.add('text-accent');}
    }
  }
  await delay(800);

  // Show before/after missing charts
  showOp('prepOp2');
  mkBar('missingBeforeChart',viz.missing_before,'Before',C.amber,C.amberFill);
  mkBar('missingAfterChart',viz.missing_after,'After',C.accent,C.accentFill);
  await delay(1200);

  // Show raw vs clean tables
  showOp('prepOp3');
  renderTable('rawTableWrap',rawSample,true);
  renderTable('cleanTableWrap',cleanSample,false);
}

function mkBar(id,data,label,bc,bg){
  if(!data||!data.length)return;destroyChart(id);
  charts[id]=new Chart($(id).getContext('2d'),{type:'bar',data:{labels:data.map(d=>d.feature),datasets:[{label,data:data.map(d=>d.missing_pct),backgroundColor:bg,borderColor:bc,borderWidth:1}]},options:{...chartOpts(),plugins:{legend:{display:false}},scales:{...chartOpts().scales,x:{...chartOpts().scales.x,ticks:{...chartOpts().scales.x.ticks,maxRotation:55,minRotation:45}}}}});
}

function renderTable(wId,rows,highlightNulls){
  const w=$(wId);if(!rows||!rows.length){w.innerHTML='<p class="muted" style="font-size:11px;">No data</p>';return;}
  const k=Object.keys(rows[0]);
  w.innerHTML=`<table class="data-table"><thead><tr>${k.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>'<tr>'+k.map(c=>{const v=r[c];if(v==null)return highlightNulls?'<td class="null-val">NaN</td>':'<td>0.00</td>';return`<td>${typeof v==='number'?fmt(v,2):v}</td>`;}).join('')+'</tr>').join('')}</tbody></table>`;
}

/* ================================================================
   STAGE 2: FEATURES — Animated counters + sliding window canvas
   ================================================================ */
async function renderFeatures(P){
  const trace=P.preprocessing_trace||[];
  const feStep=(trace).find(t=>t.step==='Feature engineering')||{};
  const scStep=(trace).find(t=>t.step==='Scaling and clipping')||{};
  const wiStep=(trace).find(t=>t.step==='Windowing')||{};
  const shape=P.stage_viz?.input_window_shape||[];

  // Stat cards with animated counters
  $('featOp1').innerHTML=`<div class="stat-row">
    <div class="stat-card"><div class="stat-value" data-t="${feStep.features_created||shape[2]||0}">0</div><div class="stat-label">Features Created</div></div>
    <div class="stat-card"><div class="stat-value" data-t="${scStep.rows||0}">0</div><div class="stat-label">Rows Scaled</div></div>
    <div class="stat-card"><div class="stat-value" data-t="${wiStep.window_count||shape[0]||0}">0</div><div class="stat-label">Windows</div></div>
    <div class="stat-card"><div class="stat-value" data-t="${wiStep.window_length||shape[1]||0}">0</div><div class="stat-label">Timesteps/Window</div></div>
  </div>`;
  showOp('featOp1');
  $('featOp1').querySelectorAll('.stat-value').forEach(el=>animateCounter(el,parseInt(el.dataset.t)||0,1800));
  await delay(2200);

  // Sliding window canvas animation
  showOp('featOp2');
  const nRows=scStep.rows||60;
  const seqLen=wiStep.window_length||shape[1]||48;
  const stride=wiStep.stride||12;
  const nWin=wiStep.window_count||shape[0]||1;
  $('windowCaption').textContent=`${nRows} rows → ${nWin} windows of ${seqLen} timesteps (stride ${stride})`;
  drawSlidingWindow('windowCanvas',nRows,seqLen,stride,nWin);
  await delay(3000);

  // Tensor shape
  $('tensorShapeDisplay').innerHTML=`<div class="shape-text">[${shape.join(' × ')||'—'}]</div><div class="shape-labels">(windows × timesteps × features) → input to TCN</div>`;
  showOp('featOp3');
}

function animateCounter(el,target,dur=1500){
  const start=performance.now();
  (function tick(now){const p=Math.min((now-start)/dur,1);el.textContent=Math.round(target*(1-Math.pow(1-p,3)));if(p<1)requestAnimationFrame(tick);else el.textContent=target;})(performance.now());
}

function drawSlidingWindow(canvasId,nRows,seqLen,stride,nWin){
  const cv=$(canvasId);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth-40;
  cv.width=W;cv.height=200;
  const rowH=4;const dataH=Math.min(nRows*rowH,160);const topY=20;
  const barW=Math.max(W-80,200);const x0=40;

  // Draw data column
  for(let i=0;i<Math.min(nRows,40);i++){
    ctx.fillStyle=`hsl(${160+i*2},40%,${25+Math.random()*10}%)`;
    ctx.fillRect(x0,topY+i*(dataH/Math.min(nRows,40)),barW,Math.max(dataH/Math.min(nRows,40)-1,2));
  }
  ctx.fillStyle='#7a9aab';ctx.font='11px Inter';ctx.fillText(`${nRows} rows`,x0,topY+dataH+16);

  // Animate sliding window
  let winIdx=0;
  const maxVisible=Math.min(nRows,40);
  const rowUnit=dataH/maxVisible;
  const winH=Math.min(seqLen/nRows*dataH,dataH);

  function animateWin(){
    // Clear overlay
    ctx.clearRect(x0-2,topY-2,barW+4,dataH+4);
    // Redraw data
    for(let i=0;i<maxVisible;i++){
      ctx.fillStyle=`hsl(${160+i*2},40%,${25+Math.random()*3}%)`;
      ctx.fillRect(x0,topY+i*rowUnit,barW,Math.max(rowUnit-1,2));
    }
    // Draw window overlay
    const winStart=(winIdx*stride/nRows)*dataH;
    ctx.strokeStyle=C.accent;ctx.lineWidth=2;
    ctx.strokeRect(x0,topY+winStart,barW,winH);
    ctx.fillStyle='rgba(16,185,129,0.15)';
    ctx.fillRect(x0,topY+winStart,barW,winH);
    // Label
    ctx.fillStyle='#7a9aab';ctx.font='11px Inter';
    ctx.clearRect(x0,topY+dataH+6,barW,20);
    ctx.fillText(`Window ${winIdx+1}/${nWin}  ·  rows ${winIdx*stride}–${Math.min(winIdx*stride+seqLen,nRows)}`,x0,topY+dataH+16);
    // Bracket on right
    ctx.fillStyle=C.accent;ctx.font='bold 11px JetBrains Mono';
    ctx.fillText(`W${winIdx+1}`,x0+barW+8,topY+winStart+winH/2+4);

    winIdx++;
    if(winIdx<nWin)setTimeout(animateWin,400);
  }
  setTimeout(animateWin,300);
}

/* ================================================================
   STAGE 3: TCN — Animated convolution diagram on canvas
   ================================================================ */
async function renderTCN(P){
  const sv=P.stage_viz||{};
  const tcn=sv.tcn||{};const tIn=sv.transformer_in||{};
  const inShape=sv.input_window_shape||[];

  showOp('tcnOp1');
  await delay(400);
  drawTCNDiagram('tcnCanvas',inShape,tcn.shape||[],tIn.shape||[]);
  await delay(3000);

  // Stats
  $('tcnStats').innerHTML=`
    <div class="glass-card"><h3>TCN Output</h3><p class="text-mono" style="font-size:18px;color:var(--accent);margin-bottom:10px;">[${(tcn.shape||[]).join(' × ')}]</p>
      <div class="tensor-card"><div class="tensor-stat"><span class="label">Mean</span><span class="val">${fmt(tcn.mean,4)}</span></div><div class="tensor-stat"><span class="label">Std</span><span class="val">${fmt(tcn.std,4)}</span></div><div class="tensor-stat"><span class="label">Min</span><span class="val">${fmt(tcn.min,4)}</span></div><div class="tensor-stat"><span class="label">Max</span><span class="val">${fmt(tcn.max,4)}</span></div></div></div>
    <div class="glass-card"><h3>→ Transformer Input</h3><p class="text-mono" style="font-size:18px;color:var(--blue);margin-bottom:10px;">[${(tIn.shape||[]).join(' × ')}]</p>
      <div class="tensor-card"><div class="tensor-stat"><span class="label">Mean</span><span class="val">${fmt(tIn.mean,4)}</span></div><div class="tensor-stat"><span class="label">Std</span><span class="val">${fmt(tIn.std,4)}</span></div><div class="tensor-stat"><span class="label">Min</span><span class="val">${fmt(tIn.min,4)}</span></div><div class="tensor-stat"><span class="label">Max</span><span class="val">${fmt(tIn.max,4)}</span></div></div></div>`;
  showOp('tcnOp2');
}

function drawTCNDiagram(canvasId,inShape,tcnShape,tInShape){
  const cv=$(canvasId);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth-40;
  cv.width=W;cv.height=280;
  const cx=W/2;

  // Input block
  drawBlock(ctx,40,30,100,60,'Input',`[${inShape.join('×')}]`,'#f59e0b');

  // Three parallel branches
  const kernels=[{k:2,d:1,label:'Kernel 2\nDilation 1'},{k:3,d:2,label:'Kernel 3\nDilation 2'},{k:5,d:4,label:'Kernel 5\nDilation 4'}];
  const branchX=[W*0.15, W*0.45, W*0.75];

  // Animate branches appearing
  let branchIdx=0;
  function drawBranch(){
    if(branchIdx>=3)return;
    const bx=branchX[branchIdx];
    const k=kernels[branchIdx];
    // Arrow from input to branch
    drawArrow(ctx,90,90,bx+45,120,'#4a6a7a');
    // Conv block
    drawBlock(ctx,bx,120,90,50,'Conv1d',`k=${k.k}, d=${k.d}`,'#3b82f6');
    // BN+GELU block
    drawBlock(ctx,bx,180,90,35,'BN → GELU','','#6366f1');
    drawArrow(ctx,bx+45,170,bx+45,180,'#4a6a7a');
    branchIdx++;
    if(branchIdx<3)setTimeout(drawBranch,500);
    else setTimeout(drawSum,500);
  }

  function drawSum(){
    // Sum node
    const sumX=cx-25,sumY=230;
    for(let i=0;i<3;i++){drawArrow(ctx,branchX[i]+45,215,sumX+25,sumY,'#4a6a7a');}
    ctx.beginPath();ctx.arc(sumX+25,sumY+15,18,0,Math.PI*2);ctx.fillStyle='rgba(16,185,129,0.2)';ctx.fill();ctx.strokeStyle=C.accent;ctx.lineWidth=2;ctx.stroke();
    ctx.fillStyle='#fff';ctx.font='bold 16px Inter';ctx.textAlign='center';ctx.fillText('Σ',sumX+25,sumY+20);
    // Output
    drawArrow(ctx,sumX+25,sumY+33,sumX+25,sumY+45,'#4a6a7a');
    drawBlock(ctx,sumX-20,sumY+45,90,30,'TCN Out',`[${tcnShape.join('×')}]`,'#10b981');
    ctx.textAlign='left';
  }

  setTimeout(drawBranch,400);
}

function drawBlock(ctx,x,y,w,h,title,sub,color){
  ctx.fillStyle=`${color}22`;ctx.strokeStyle=color;ctx.lineWidth=1.5;
  roundRect(ctx,x,y,w,h,8);ctx.fill();ctx.stroke();
  ctx.fillStyle='#e4eff2';ctx.font='bold 11px Inter';ctx.textAlign='center';
  ctx.fillText(title,x+w/2,y+h/2+(sub?-4:4));
  if(sub){ctx.fillStyle='#7a9aab';ctx.font='10px JetBrains Mono';ctx.fillText(sub,x+w/2,y+h/2+10);}
  ctx.textAlign='left';
}

function drawArrow(ctx,x1,y1,x2,y2,color){
  ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();
  const angle=Math.atan2(y2-y1,x2-x1);const hs=6;
  ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-hs*Math.cos(angle-0.4),y2-hs*Math.sin(angle-0.4));ctx.lineTo(x2-hs*Math.cos(angle+0.4),y2-hs*Math.sin(angle+0.4));ctx.closePath();ctx.fillStyle=color;ctx.fill();
}

function roundRect(ctx,x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();}

/* ================================================================
   STAGE 4: TRANSFORMER — Attention diagram + heatmap
   ================================================================ */
async function renderTransformer(P){
  const sv=P.stage_viz||{};
  const tOut=sv.transformer_out||{};const attn=sv.attention_proxy||[];

  showOp('transOp1');
  await delay(400);
  drawAttentionDiagram('attnDiagramCanvas',sv);
  if(attn.length)animateHeatmap('heatmapCanvas',attn);
  await delay(3500);

  // Stats
  $('transStats').innerHTML=`
    <div class="glass-card"><h3>Transformer Output</h3><p class="text-mono" style="font-size:18px;color:var(--blue);margin-bottom:10px;">[${(tOut.shape||[]).join(' × ')}]</p>
      <div class="tensor-card"><div class="tensor-stat"><span class="label">Mean</span><span class="val">${fmt(tOut.mean,4)}</span></div><div class="tensor-stat"><span class="label">Std</span><span class="val">${fmt(tOut.std,4)}</span></div><div class="tensor-stat"><span class="label">Min</span><span class="val">${fmt(tOut.min,4)}</span></div><div class="tensor-stat"><span class="label">Max</span><span class="val">${fmt(tOut.max,4)}</span></div></div></div>
    <div class="glass-card"><h3>Classification Head</h3><p class="muted" style="margin-bottom:8px;">Last timestep → Linear → Sigmoid → Probability</p>
      <div class="tensor-card"><div class="tensor-stat"><span class="label">Input</span><span class="val text-blue">[1×d]</span></div><div class="tensor-stat"><span class="label">Output</span><span class="val text-accent">p ∈ [0,1]</span></div></div></div>`;
  showOp('transOp2');
}

function drawAttentionDiagram(canvasId,sv){
  const cv=$(canvasId);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth-40;
  cv.width=W;cv.height=260;
  const tIn=sv.transformer_in||{};
  const tOut=sv.transformer_out||{};
  const nTokens=Math.min((tIn.shape||[])[1]||8,12);

  // Draw tokens as circles at top
  const tokenY=40;const spacing=Math.min(W/(nTokens+2),50);const startX=(W-nTokens*spacing)/2+spacing/2;

  // Draw tokens
  for(let i=0;i<nTokens;i++){
    const x=startX+i*spacing;
    ctx.beginPath();ctx.arc(x,tokenY,12,0,Math.PI*2);ctx.fillStyle='rgba(59,130,246,0.3)';ctx.fill();ctx.strokeStyle=C.blue;ctx.lineWidth=1.5;ctx.stroke();
    ctx.fillStyle='#e4eff2';ctx.font='bold 9px JetBrains Mono';ctx.textAlign='center';ctx.fillText(`t${i+1}`,x,tokenY+3);
  }
  ctx.textAlign='left';ctx.fillStyle='#7a9aab';ctx.font='11px Inter';ctx.fillText('Input tokens (timesteps)',startX-10,tokenY-22);

  // Animate attention arcs between tokens
  let arcIdx=0;const totalArcs=Math.min(nTokens*3,30);
  function drawArc(){
    if(arcIdx>=totalArcs)return;
    const from=Math.floor(Math.random()*nTokens);
    let to=Math.floor(Math.random()*nTokens);if(to===from)to=(from+1)%nTokens;
    const x1=startX+from*spacing,x2=startX+to*spacing;
    const midY=tokenY+40+Math.random()*60;
    const alpha=0.15+Math.random()*0.25;
    ctx.beginPath();ctx.moveTo(x1,tokenY+12);ctx.quadraticCurveTo((x1+x2)/2,midY,x2,tokenY+12);
    ctx.strokeStyle=`rgba(16,185,129,${alpha})`;ctx.lineWidth=1+Math.random()*2;ctx.stroke();
    arcIdx++;
    setTimeout(drawArc,80);
  }
  setTimeout(drawArc,300);

  // Labels
  setTimeout(()=>{
    ctx.fillStyle='#7a9aab';ctx.font='11px Inter';ctx.textAlign='center';
    ctx.fillText('Self-Attention: each token attends to all others',W/2,160);
    drawBlock(ctx,W/2-60,175,120,30,'Output',`[${(tOut.shape||[]).join('×')}]`,'#10b981');
    ctx.textAlign='left';
  },totalArcs*80+200);
}

function animateHeatmap(cid,matrix){
  const cv=$(cid);if(!cv)return;
  const ctx=cv.getContext('2d'),n=matrix.length;
  const size=Math.min(320,(cv.parentElement?.clientWidth||320)-20);
  cv.width=size;cv.height=size;const cs=size/n;let progress=0;
  (function draw(){progress+=0.015;if(progress>1)progress=1;
    const cells=Math.floor(n*n*progress);
    for(let idx=0;idx<cells;idx++){const i=Math.floor(idx/n),j=idx%n,v=matrix[i][j];
      ctx.fillStyle=`rgb(${Math.round(v*240)},${Math.round(80+v*120)},${Math.round(60+(1-v)*100)})`;
      ctx.fillRect(j*cs,i*cs,cs+0.5,cs+0.5);}
    if(progress<1)requestAnimationFrame(draw);})();
}

/* ================================================================
   STAGE 5: OUTPUT — Sigmoid visualization + results
   ================================================================ */
async function renderOutput(P){
  const risk=P.risk_score,thr=P.threshold,pred=P.prediction,label=P.label,conf=P.confidence;

  // Sigmoid canvas
  showOp('outOp0');
  drawSigmoid('sigmoidCanvas',risk,thr);
  await delay(2000);

  // Banner
  const banner=$('predictionBanner');
  banner.className=`prediction-banner ${pred===1?'high-risk':'low-risk'}`;
  banner.innerHTML=`<div class="pred-label">${label}</div><div class="pred-stats"><div class="pred-stat"><span>Risk Score</span><strong>${fmt(risk,4)}</strong></div><div class="pred-stat"><span>Confidence</span><strong>${fmt(conf*100,1)}%</strong></div><div class="pred-stat"><span>Threshold</span><strong>${fmt(thr,3)}</strong></div><div class="pred-stat"><span>Windows</span><strong>${P.window_count}</strong></div></div>`;
  requestAnimationFrame(()=>banner.classList.add('visible'));
  await delay(1000);

  // Charts
  showOp('outOp1');
  $('riskGaugeWrap').innerHTML=`<div class="risk-gauge"><div class="risk-bar-track"><div class="risk-bar-fill" id="riskBarFill"></div></div><div class="risk-bar-label"><span>0.0 (Low)</span><span>Thr: ${fmt(thr,3)}</span><span>1.0 (High)</span></div></div>`;
  requestAnimationFrame(()=>requestAnimationFrame(()=>{$('riskBarFill').style.width=Math.max(0,Math.min(100,risk*100))+'%';}));

  const probs=P.window_probabilities||[];
  destroyChart('probChart');
  charts['probChart']=new Chart($('probChart').getContext('2d'),{type:'line',data:{labels:probs.map((_,i)=>`W${i+1}`),datasets:[{label:'Window Risk',data:probs,borderColor:C.accent,backgroundColor:C.accentFill,pointRadius:3,fill:true,tension:0.3},{label:'Threshold',data:probs.map(()=>thr),borderColor:C.red,borderDash:[6,6],pointRadius:0,fill:false}]},options:{...chartOpts(),scales:{...chartOpts().scales,y:{...chartOpts().scales.y,min:0,max:1}}}});

  await delay(1200);showOp('outOp2');
  let html='';const cm=P.confusion_matrix;
  if(cm){const m=cm.matrix;html+=`<div class="glass-card"><h3>Confusion Matrix</h3><div class="confusion-grid"><div class="confusion-cell header"></div><div class="confusion-cell header">Pred Non-Sepsis</div><div class="confusion-cell header">Pred Sepsis</div><div class="confusion-cell row-label">Actual Non-Sepsis</div><div class="confusion-cell tn">${m[0][0]}</div><div class="confusion-cell fp">${m[0][1]}</div><div class="confusion-cell row-label">Actual Sepsis</div><div class="confusion-cell fn">${m[1][0]}</div><div class="confusion-cell tp">${m[1][1]}</div></div></div>`;}
  if(P.roc_curve?.length)html+=`<div class="glass-card"><h3>ROC Curve${P.uploaded_metrics?.auroc?' (AUROC: '+fmt(P.uploaded_metrics.auroc,3)+')':''}</h3><div class="chart-container"><canvas id="rocChart"></canvas></div></div>`;
  if(P.pr_curve?.length)html+=`<div class="glass-card"><h3>PR Curve${P.uploaded_metrics?.auprc?' (AUPRC: '+fmt(P.uploaded_metrics.auprc,3)+')':''}</h3><div class="chart-container"><canvas id="prChart"></canvas></div></div>`;
  const um=P.uploaded_metrics;
  if(um)html+=`<div class="glass-card"><h3>Evaluation Metrics</h3><p class="muted" style="font-size:11px;margin-bottom:8px;">${P.uploaded_metrics_note||um.sample_count+' windows'}</p><div class="tensor-card"><div class="tensor-stat"><span class="label">Accuracy</span><span class="val">${fmt(um.accuracy,3)}</span></div><div class="tensor-stat"><span class="label">Precision</span><span class="val">${fmt(um.precision,3)}</span></div><div class="tensor-stat"><span class="label">Recall</span><span class="val">${fmt(um.recall,3)}</span></div><div class="tensor-stat"><span class="label">F1</span><span class="val">${fmt(um.f1,3)}</span></div></div></div>`;
  else if(P.uploaded_metrics_note)html+=`<div class="glass-card"><h3>Note</h3><p class="muted">${P.uploaded_metrics_note}</p></div>`;
  $('metricsRow').innerHTML=html;

  requestAnimationFrame(()=>{
    if(P.roc_curve?.length){const c=$('rocChart');if(c){destroyChart('rocChart');charts['rocChart']=new Chart(c.getContext('2d'),{type:'line',data:{datasets:[{label:'ROC',data:P.roc_curve.map(p=>({x:p.x,y:p.y})),borderColor:C.accent,backgroundColor:C.accentFill,pointRadius:0,fill:true,tension:0.2},{label:'Random',data:[{x:0,y:0},{x:1,y:1}],borderColor:C.tick,borderDash:[4,4],pointRadius:0,fill:false}]},options:{...chartOpts(),scales:{x:{type:'linear',min:0,max:1,title:{display:true,text:'FPR',color:C.tick},ticks:{color:C.tick},grid:{color:C.grid}},y:{type:'linear',min:0,max:1,title:{display:true,text:'TPR',color:C.tick},ticks:{color:C.tick},grid:{color:C.grid}}}}});}}
    if(P.pr_curve?.length){const c=$('prChart');if(c){destroyChart('prChart');charts['prChart']=new Chart(c.getContext('2d'),{type:'line',data:{datasets:[{label:'PR',data:P.pr_curve.map(p=>({x:p.x,y:p.y})),borderColor:C.blue,backgroundColor:'rgba(59,130,246,0.1)',pointRadius:0,fill:true,tension:0.2}]},options:{...chartOpts(),scales:{x:{type:'linear',min:0,max:1,title:{display:true,text:'Recall',color:C.tick},ticks:{color:C.tick},grid:{color:C.grid}},y:{type:'linear',min:0,max:1,title:{display:true,text:'Precision',color:C.tick},ticks:{color:C.tick},grid:{color:C.grid}}}}});}}
  });
}

function drawSigmoid(canvasId,risk,thr){
  const cv=$(canvasId);if(!cv)return;
  const ctx=cv.getContext('2d');
  const W=cv.parentElement.clientWidth-40;
  cv.width=W;cv.height=200;
  const px=60,py=20,pw=W-120,ph=150;
  // Axes
  ctx.strokeStyle='#2a3a4a';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px,py+ph);ctx.lineTo(px+pw,py+ph);ctx.stroke();
  ctx.fillStyle='#7a9aab';ctx.font='10px Inter';ctx.textAlign='right';
  ctx.fillText('1.0',px-8,py+8);ctx.fillText('0.5',px-8,py+ph/2+4);ctx.fillText('0.0',px-8,py+ph+4);
  ctx.textAlign='center';ctx.fillText('Logit →',px+pw/2,py+ph+16);

  // Sigmoid curve animated
  let progress=0;
  function drawCurve(){
    progress+=0.03;if(progress>1)progress=1;
    const nPts=Math.floor(200*progress);
    ctx.beginPath();
    for(let i=0;i<=nPts;i++){
      const t=i/200;const logit=(t-0.5)*12;
      const sig=1/(1+Math.exp(-logit));
      const x=px+t*pw,y=py+ph-sig*ph;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.strokeStyle=C.accent;ctx.lineWidth=2.5;ctx.stroke();

    if(progress>=1){
      // Threshold line
      const thrY=py+ph-thr*ph;
      ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(px,thrY);ctx.lineTo(px+pw,thrY);
      ctx.strokeStyle=C.red;ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle=C.red;ctx.font='bold 11px Inter';ctx.textAlign='left';
      ctx.fillText(`Threshold = ${fmt(thr,3)}`,px+pw+4,thrY+4);

      // Risk marker animated
      let markerProgress=0;
      function drawMarker(){
        markerProgress+=0.04;if(markerProgress>1)markerProgress=1;
        const currentRisk=risk*markerProgress;
        const markerY=py+ph-currentRisk*ph;
        // Clear marker area
        ctx.clearRect(px-5,py-5,pw+80,ph+30);
        // Redraw axes + curve
        ctx.strokeStyle='#2a3a4a';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px,py+ph);ctx.lineTo(px+pw,py+ph);ctx.stroke();
        ctx.fillStyle='#7a9aab';ctx.font='10px Inter';ctx.textAlign='right';ctx.fillText('1.0',px-8,py+8);ctx.fillText('0.5',px-8,py+ph/2+4);ctx.fillText('0.0',px-8,py+ph+4);
        ctx.textAlign='center';ctx.fillText('Logit →',px+pw/2,py+ph+16);
        // Curve
        ctx.beginPath();
        for(let i=0;i<=200;i++){const t=i/200;const logit=(t-0.5)*12;const sig=1/(1+Math.exp(-logit));const x=px+t*pw,y=py+ph-sig*ph;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
        ctx.strokeStyle=C.accent;ctx.lineWidth=2.5;ctx.stroke();
        // Threshold
        const thrY2=py+ph-thr*ph;
        ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(px,thrY2);ctx.lineTo(px+pw,thrY2);ctx.strokeStyle=C.red;ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle=C.red;ctx.font='bold 11px Inter';ctx.textAlign='left';ctx.fillText(`Threshold = ${fmt(thr,3)}`,px+pw+4,thrY2+4);
        // Marker
        ctx.beginPath();ctx.arc(px+pw*0.7,markerY,7,0,Math.PI*2);
        ctx.fillStyle=risk>=thr?C.red:C.accent;ctx.fill();
        ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
        // Horizontal line from marker
        ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(px,markerY);ctx.lineTo(px+pw*0.7,markerY);ctx.strokeStyle='rgba(255,255,255,0.2)';ctx.lineWidth=1;ctx.stroke();ctx.setLineDash([]);
        // Value label
        ctx.fillStyle='#e4eff2';ctx.font='bold 12px JetBrains Mono';ctx.textAlign='left';
        ctx.fillText(`p = ${fmt(currentRisk,4)}`,px+pw*0.7+14,markerY+4);

        if(markerProgress<1)requestAnimationFrame(drawMarker);
      }
      requestAnimationFrame(drawMarker);
    }else{requestAnimationFrame(drawCurve);}
  }
  requestAnimationFrame(drawCurve);
}

/* ===== INIT ===== */
document.addEventListener('DOMContentLoaded',()=>{initParticles();loadOverview();initUpload();initNavButtons();setTracker(0);});
