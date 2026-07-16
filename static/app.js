let W=0,H=0,ZOOM=2,tool="paint",brush=14,opacity=.45,showNod=true,showLung=true,layer="lung";
let col0=[255,85,85],col1=[0,230,90],nLayers=2;   // set from /api/config
let samInfo={lung:{available:false,backend:"",prompt:"point"},nodule:{available:false,backend:"",prompt:"box"}};
let mask=null,nodule=null,pids=[],editedArr=[],cur=0,undoStk=[],redoStk=[],chestImg=new Image(),dirty=false;
const curArr=()=>layer==="lung"?mask:nodule;
function updateProg(){const done=editedArr.filter(Boolean).length,t=pids.length||1;
  $("progfill").style.width=(100*done/t)+"%";
  $("prog").textContent=done+" / "+pids.length+" edited · "+(pids.length-done)+" remaining";}
const renderCur=()=>layer==="lung"?renderMask():renderNodule();
const bg=document.getElementById("bg"),mk=document.getElementById("mk"),ov=document.getElementById("ov");
const stage=document.getElementById("stage"),status=document.getElementById("status");
const $=id=>document.getElementById(id);

function setStatus(t,c){status.textContent=t;status.style.color=c||"#9cdcfe";}
async function loadPids(){
  const r=await (await fetch("/api/pids")).json();
  pids=r.pids; editedArr=r.edited||pids.map(()=>false); const sel=$("pid"); sel.innerHTML="";
  pids.forEach(p=>{const o=document.createElement("option");o.value=o.textContent=p;sel.appendChild(o);});
  updateProg();
  if(pids.length){let i=editedArr.findIndex(v=>!v); cur=i<0?0:i; await loadSlice(pids[cur]);}
  else setStatus("No series found — check --series/--dataset","#f88");
}
function pngToMask(url,cb){const im=new Image();im.onload=()=>{
  const c=document.createElement("canvas");c.width=W;c.height=H;const x=c.getContext("2d");
  x.drawImage(im,0,0);const d=x.getImageData(0,0,W,H).data;const m=new Uint8Array(W*H);
  for(let i=0;i<W*H;i++)m[i]=d[i*4]>127?1:0;cb(m);};im.src=url;}
async function loadSlice(pid){
  setStatus("loading "+pid+"…");
  const r=await (await fetch("/api/slice/"+pid)).json();
  W=r.w;H=r.h;
  for(const c of [bg,mk,ov]){c.width=W;c.height=H;}
  applyZoom();
  chestImg.onload=()=>{bg.getContext("2d").drawImage(chestImg,0,0);};
  chestImg.src=r.chest;
  await new Promise(res=>pngToMask(r.mask,m=>{mask=m;res();}));
  await new Promise(res=>pngToMask(r.nodule,m=>{nodule=m;res();}));
  undoStk=[];redoStk=[];dirty=false;renderMask();renderNodule();
  $("pill").textContent=(cur+1)+"/"+pids.length+(r.edited?" ·edited":"");
  $("pid").value=pid; setStatus("loaded "+pid, r.edited?"#dcdcaa":"#9cdcfe");
}
function applyZoom(){for(const c of [bg,mk,ov]){c.style.width=(W*ZOOM)+"px";c.style.height=(H*ZOOM)+"px";}
  stage.style.width=(W*ZOOM)+"px";stage.style.height=(H*ZOOM)+"px";}
// red = editable lung-parenchyma mask
// layer 0 (default red)
function renderMask(){const x=mk.getContext("2d");const img=x.createImageData(W,H);
  if(showLung&&mask){const a=Math.round(opacity*255);for(let i=0;i<W*H;i++){if(mask[i]){
    img.data[i*4]=col0[0];img.data[i*4+1]=col0[1];img.data[i*4+2]=col0[2];img.data[i*4+3]=a;}}}
  x.putImageData(img,0,0);}
// layer 1 (default green)
function renderNodule(){const x=ov.getContext("2d");const img=x.createImageData(W,H);
  if(showNod&&nodule){const a=Math.round(Math.min(1,opacity+0.25)*255);
    for(let i=0;i<W*H;i++){if(nodule[i]){
      img.data[i*4]=col1[0];img.data[i*4+1]=col1[1];img.data[i*4+2]=col1[2];img.data[i*4+3]=a;}}}
  x.putImageData(img,0,0);}
function pushUndo(){undoStk.push({m:mask.slice(),n:nodule.slice()});
  if(undoStk.length>40)undoStk.shift();redoStk=[];dirty=true;}
function paintAt(cx,cy,add){const arr=curArr(),r=brush/2,r2=r*r;
  const x0=Math.max(0,Math.floor(cx-r)),x1=Math.min(W-1,Math.ceil(cx+r));
  const y0=Math.max(0,Math.floor(cy-r)),y1=Math.min(H-1,Math.ceil(cy+r));
  for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){const dx=x-cx,dy=y-cy;
    if(dx*dx+dy*dy<=r2)arr[y*W+x]=add?1:0;}renderCur();}
let drawing=false,addMode=true,samDrag=null;
const boxEl=$("sambox");
function evtXY(e){const rc=mk.getBoundingClientRect();
  return [(e.clientX-rc.left)/ZOOM,(e.clientY-rc.top)/ZOOM];}
// brush-size cursor ring, coloured by the active layer (points 1 & 9)
const cursorEl=$("cursor");
const rgb=a=>`rgb(${a[0]},${a[1]},${a[2]})`;
const layerColor=()=>layer==="lung"?col0:col1;
function sizeCursor(){const d=Math.max(6,brush*ZOOM);
  cursorEl.style.width=d+"px";cursorEl.style.height=d+"px";cursorEl.style.borderColor=rgb(layerColor());}
function moveCursor(e){cursorEl.style.left=e.clientX+"px";cursorEl.style.top=e.clientY+"px";}
function tintStage(){stage.style.boxShadow="0 0 0 2px "+rgb(layerColor())+",0 10px 40px rgba(0,0,0,.6)";}
mk_events();
function mk_events(){
 ov.addEventListener("contextmenu",e=>e.preventDefault());
 ov.addEventListener("mousedown",e=>{e.preventDefault();const[x,y]=evtXY(e);
   if(tool==="sam"&&e.button!==2){          // SAM: drag = box prompt, plain click = point prompt
     samDrag={x0:x,y0:y,cx:e.clientX,cy:e.clientY};
     boxEl.style.display="block";boxEl.style.left=e.clientX+"px";boxEl.style.top=e.clientY+"px";
     boxEl.style.width="0px";boxEl.style.height="0px";return;}
   pushUndo();drawing=true;
   addMode=!(e.button===2||tool==="erase");paintAt(x,y,addMode);});
 window.addEventListener("mousemove",e=>{
   if(samDrag){const L=Math.min(e.clientX,samDrag.cx),T=Math.min(e.clientY,samDrag.cy);
     boxEl.style.left=L+"px";boxEl.style.top=T+"px";
     boxEl.style.width=Math.abs(e.clientX-samDrag.cx)+"px";boxEl.style.height=Math.abs(e.clientY-samDrag.cy)+"px";return;}
   if(!drawing)return;const[x,y]=evtXY(e);paintAt(x,y,addMode);});
 window.addEventListener("mouseup",e=>{
   if(samDrag){boxEl.style.display="none";
     const moved=Math.abs(e.clientX-samDrag.cx)+Math.abs(e.clientY-samDrag.cy);
     const[x,y]=evtXY(e);
     if(moved<6) samClick(samDrag.x0,samDrag.y0);       // barely moved -> treat as a click
     else samBox(Math.min(samDrag.x0,x),Math.min(samDrag.y0,y),Math.max(samDrag.x0,x),Math.max(samDrag.y0,y));
     samDrag=null;return;}
   drawing=false;});
 ov.addEventListener("mousemove",moveCursor);
 ov.addEventListener("mouseenter",e=>{moveCursor(e);sizeCursor();cursorEl.style.display="block";});
 ov.addEventListener("mouseleave",()=>{cursorEl.style.display="none";});
}
// SAM: apply a returned mask to the active layer (one undo step).
// clearBox (optional [x0,y0,x1,y1]) is wiped before painting in the new mask — box prompts
// pass their own query box so a re-segmented region doesn't just get merged underneath
// whatever solid fill was already there (e.g. a seed square hiding the real segmented shape).
function applySamMask(pngUrl,how,clearBox){pushUndo();
  pngToMask(pngUrl,m=>{const arr=curArr();
    if(clearBox){
      const x0=Math.max(0,Math.floor(clearBox[0])),x1=Math.min(W-1,Math.ceil(clearBox[2]));
      const y0=Math.max(0,Math.floor(clearBox[1])),y1=Math.min(H-1,Math.ceil(clearBox[3]));
      for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++) arr[y*W+x]=0;
    }
    for(let i=0;i<W*H;i++) if(m[i]) arr[i]=1;
    renderCur();setStatus(how+" → "+layer+" — refine with the brush","#8f8");});}
async function samPrompt(body,how,clearBox){setStatus("segmenting…","#c08cff");
  let r;
  try{ r=await (await fetch("/api/segment/"+pids[cur],{method:"POST",
        headers:{"Content-Type":"application/json"},body:JSON.stringify({...body,layer})})).json();
  }catch(err){setStatus("segment request failed","#f88");return;}
  if(r.error){setStatus(r.error,"#f88");return;}
  applySamMask(r.mask,how,clearBox);}
const samClick=(x,y)=>samPrompt({x:Math.round(x),y:Math.round(y)},"click-segmented",null);
// Clear the drawn box UNIONed with whatever was already on the layer — not just the drawn
// box alone — so a fragment of an old mask sitting outside the new box (e.g. the seed
// square didn't line up with where you just dragged) doesn't survive as a stray leftover.
// The model prompt itself still only gets the box you actually drew.
const samBox=(x0,y0,x1,y1)=>{const drawn=[Math.round(x0),Math.round(y0),Math.round(x1),Math.round(y1)];
  const old=layerBBox(curArr());
  const clear=old?[Math.min(drawn[0],old[0]),Math.min(drawn[1],old[1]),
                   Math.max(drawn[2],old[2]),Math.max(drawn[3],old[3])]:drawn;
  samPrompt({box:drawn},"box-segmented",clear);};
// bounding box of what's currently painted on a layer
function layerBBox(arr){let x0=1e9,y0=1e9,x1=-1,y1=-1;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){ if(arr[y*W+x]){
    if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y; } }
  return x1<0?null:[x0,y0,x1,y1];}
// Use the layer's EXISTING mask (e.g. the nodule's square footprint) as the box prompt and
// re-segment the structure inside it — clearing just that box (not the whole layer) so the
// refined shape is actually visible instead of staying hidden under the old solid fill.
function autoSeg(){const b=layerBBox(curArr());
  if(!b){setStatus("nothing on the "+layer+" layer to take a box from","#f88");return;}
  // pad generously (not just a few px): the model was trained on boxes noticeably larger
  // than the structure, so a box barely bigger than the seed square tends to just come
  // back as "the whole box" instead of the actual shape inside it.
  const p=Math.max(12,Math.round(0.4*Math.max(b[2]-b[0],b[3]-b[1])));
  const box=[Math.max(0,b[0]-p),Math.max(0,b[1]-p),Math.min(W-1,b[2]+p),Math.min(H-1,b[3]+p)];
  samPrompt({box},"refined from its box",box);}
async function cleanOp(op){setStatus(op+" ("+layer+")…");
  const r=await (await fetch("/api/clean/"+op,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mask:arrToPng(curArr()),w:W,h:H})})).json();
  if(r.error){setStatus(r.error,"#f88");return;}
  pushUndo();pngToMask(r.mask,m=>{if(layer==="lung")mask=m;else nodule=m;renderCur();setStatus(op+" done");});}
function arrToPng(arr){const c=document.createElement("canvas");c.width=W;c.height=H;const x=c.getContext("2d");
  const img=x.createImageData(W,H);for(let i=0;i<W*H;i++){const v=arr[i]?255:0;
    img.data[i*4]=img.data[i*4+1]=img.data[i*4+2]=v;img.data[i*4+3]=255;}
  x.putImageData(img,0,0);return c.toDataURL("image/png");}
async function save(){setStatus("saving…");
  const r=await (await fetch("/api/save/"+pids[cur],{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mask:arrToPng(mask),nodule:arrToPng(nodule),w:W,h:H})})).json();
  if(r.ok){dirty=false;editedArr[cur]=true;updateProg();$("pill").textContent=(cur+1)+"/"+pids.length+" ·edited";}
  setStatus(r.ok?("saved → "+r.mask_path):"save failed", r.ok?"#8f8":"#f88");
  return r.ok;}
function nextUnedited(){for(let k=1;k<=pids.length;k++){const i=(cur+k)%pids.length;
  if(!editedArr[i]){gotoPid(i);return;}}setStatus("all series edited 🎉","#8f8");}
// SAM button reflects the active layer's model, plus a two-stage toggle for nodule:
// first click arms the tool (drag a box, or click for the existing-box shortcut);
// once armed, the label switches to "Automatic segmentation" and a second click on the
// button itself (not the canvas) re-segments from whatever's already on the layer.
function updateSamButton(){
  if(layer==="nodule"&&tool==="sam"){$("sam").textContent="⚡ Automatic segmentation";return;}
  const name={mobilesam:"MobileSAM",litemedsam:"MedSAM"}[(samInfo[layer]||{}).backend]||"SAM";
  $("sam").textContent="✨ "+name;
}
function applySamVisibility(){
  const on=!!(samInfo[layer]&&samInfo[layer].available);
  $("sam").style.display=on?"":"none";$("autoseg").style.display=on?"":"none";
  updateSamButton();
}
function setLayer(l){layer=l;
  $("lyLung").classList.toggle("sec",l!=="lung");$("lyNod").classList.toggle("sec",l!=="nodule");
  if(l==="nodule"&&!showNod){showNod=true;$("nod").checked=true;renderNodule();}
  if(l==="lung"&&!showLung){showLung=true;$("lungvis").checked=true;renderMask();}
  sizeCursor();tintStage();applySamVisibility();
  setStatus("editing "+l+" layer",l==="nodule"?"#00e65a":"#ff8a8a");}
// auto-save the current edits before moving to another image
async function gotoPid(i){if(i<0||i>=pids.length||i===cur)return;
  if(dirty)await save(); cur=i; await loadSlice(pids[cur]);}
// controls
$("pid").onchange=e=>gotoPid(pids.indexOf(e.target.value));
$("prev").onclick=()=>gotoPid(cur-1);
$("next").onclick=()=>gotoPid(cur+1);
$("nextUn").onclick=nextUnedited;
function setTool(t){tool=t;for(const id of ["paint","erase","sam"]){const b=$(id);if(b)b.classList.toggle("sec",id!==t);}
  updateSamButton();
  if(t==="sam"){const mode=(samInfo[layer]||{}).prompt||"point";
    setStatus(mode==="box" ? "drag a box around the structure ("+layer+")"
                            : "click a structure ("+layer+"); drag = box","#c08cff");}}
$("paint").onclick=()=>setTool("paint");
$("erase").onclick=()=>setTool("erase");
$("sam").onclick=()=>{
  if(layer==="nodule"&&tool==="sam") autoSeg();  // armed + clicked again -> use the existing box
  else setTool("sam");
};
$("brush").oninput=e=>{brush=+e.target.value;$("bs").textContent=brush;sizeCursor();};
$("op").oninput=e=>{opacity=e.target.value/100;renderMask();renderNodule();};
$("zoom").oninput=e=>{ZOOM=e.target.value/10;$("zl").textContent=ZOOM.toFixed(1);applyZoom();sizeCursor();};
$("nod").onchange=e=>{showNod=e.target.checked;renderNodule();};
$("lungvis").onchange=e=>{showLung=e.target.checked;renderMask();};
$("lyLung").onclick=()=>setLayer("lung");
$("lyNod").onclick=()=>setLayer("nodule");
$("undo").onclick=()=>{if(undoStk.length){redoStk.push({m:mask.slice(),n:nodule.slice()});
  const s=undoStk.pop();mask=s.m;nodule=s.n;renderMask();renderNodule();}};
$("redo").onclick=()=>{if(redoStk.length){undoStk.push({m:mask.slice(),n:nodule.slice()});
  const s=redoStk.pop();mask=s.m;nodule=s.n;renderMask();renderNodule();}};
$("fill").onclick=()=>cleanOp("fill");
$("islands").onclick=()=>cleanOp("islands");
$("autoseg").onclick=autoSeg;
$("save").onclick=save;
$("reset").onclick=async()=>{const r=await (await fetch("/api/slice/"+pids[cur]+"?orig=1")).json();
  pushUndo();pngToMask(r.mask,m=>{mask=m;renderMask();});pngToMask(r.nodule,m=>{nodule=m;renderNodule();});
  setStatus("reset (not yet saved)");};
document.addEventListener("keydown",e=>{
  if(e.ctrlKey&&e.key==="z"){e.preventDefault();$("undo").click();return;}
  if(e.ctrlKey&&e.key==="s"){e.preventDefault();save();return;}
  if(e.key==="1")setLayer("lung"); if(e.key==="2")setLayer("nodule");
  if(e.key==="u")nextUnedited();
  if(e.key==="h"&&!e.repeat){mk.style.opacity="0";ov.style.opacity="0";}   // hold to peek raw CT
  if(e.key==="p")$("paint").click(); if(e.key==="e")$("erase").click();
  if(e.key==="[")$("brush").value=Math.max(1,brush-2),$("brush").oninput({target:$("brush")});
  if(e.key==="]")$("brush").value=Math.min(60,brush+2),$("brush").oninput({target:$("brush")});
  if(e.key==="ArrowLeft")$("prev").click(); if(e.key==="ArrowRight")$("next").click();
});
// apply layer names/colours/folders from the backend (re-run any time the data source changes)
function applyConfig(c){
  nLayers=c.nlayers; const L=c.layers, rgb=a=>`rgb(${a[0]},${a[1]},${a[2]})`;
  $("title").textContent=c.title||"Mask editor";
  col0=L[0].color;
  $("lyLung").textContent="● "+L[0].name; $("lyLung").style.color=rgb(col0);
  $("sw0t").textContent="Show "+L[0].name; $("dot0").style.background=rgb(col0); $("leg0t").textContent=L[0].name;
  if(nLayers>1){col1=L[1].color;
    $("lyNod").textContent="● "+L[1].name; $("lyNod").style.color=rgb(col1);
    $("sw1t").textContent="Show "+L[1].name; $("dot1").style.background=rgb(col1); $("leg1t").textContent=L[1].name;
    for(const id of ["lyNod","sw1","leg1"]) $(id).style.display="";
  }else{ // single layer: hide the second layer's controls
    for(const id of ["lyNod","sw1","leg1"]) $(id).style.display="none";
  }
  samInfo=c.sam||samInfo;
  applySamVisibility();  // shows/hides + labels #sam and #autoseg for the active layer
  tintStage();
  // prefill the splash form so it reflects what's actually loaded; clear the other mode
  if(c.mode==="generic"){
    $("tglLuna").checked=false; $("cfgDataset").value=""; $("cfgSeries").value="";
    $("cfgImages").value=c.images||"";
    if(c.masks){$("tglLung").checked=true; $("cfgMasks").value=c.masks;}
    else{$("tglLung").checked=false; $("cfgMasks").value="";}
    if(c.masks2){$("tglNodule").checked=true; $("noduleMode").value="folder";
      $("cfgMasks2").value=c.masks2; $("cfgNoduleCsv").value=""; $("cfgNoduleCols").value="";}
    else if(c.nodule_csv){$("tglNodule").checked=true; $("noduleMode").value="csv";
      $("cfgNoduleCsv").value=c.nodule_csv; $("cfgNoduleCols").value=c.nodule_cols||""; $("cfgMasks2").value="";}
    else{$("tglNodule").checked=false; $("noduleMode").value="folder";
      $("cfgMasks2").value=""; $("cfgNoduleCsv").value=""; $("cfgNoduleCols").value="";}
  }else{
    $("tglLuna").checked=true; $("cfgDataset").value=c.dataset||""; $("cfgSeries").value=c.series||"";
    $("cfgImages").value=""; $("tglLung").checked=false; $("cfgMasks").value="";
    $("tglNodule").checked=false; $("noduleMode").value="folder";
    $("cfgMasks2").value=""; $("cfgNoduleCsv").value=""; $("cfgNoduleCols").value="";
  }
  updateLungVis(); updateNoduleVis(); updateLunaVis();
  Object.assign(orig, currentFields());
}
let orig={};
async function initConfig(){
  applyConfig(await (await fetch("/api/config")).json());
  await loadPids();
}
// release the peek key -> restore the mask overlays
document.addEventListener("keyup",e=>{if(e.key==="h"){mk.style.opacity="1";ov.style.opacity="1";}});
// splash / landing page: red spotlight follows the mouse, dismiss on Start or Enter
$("splash").addEventListener("mousemove",e=>{
  const s=$("splash").style; s.setProperty("--mx",e.clientX+"px"); s.setProperty("--my",e.clientY+"px");});
function dismissSplash(){$("splash").classList.add("hidden");}
function setSplashErr(msg){$("splashErr").textContent=msg||"";}
function updateLungVis(){$("lungFields").style.display=$("tglLung").checked?"":"none";}
function updateNoduleMode(){
  const csv=$("noduleMode").value==="csv";
  $("noduleFolderField").style.display=csv?"none":"";
  $("noduleCsvField").style.display=csv?"":"none";
  $("noduleColsField").style.display=csv?"":"none";
}
function updateNoduleVis(){$("noduleFields").style.display=$("tglNodule").checked?"":"none"; updateNoduleMode();}
function updateLunaVis(){
  const on=$("tglLuna").checked;
  $("genericSection").style.display=on?"none":"";
  $("lunaFields").style.display=on?"":"none";
}
$("tglLung").onchange=updateLungVis;
$("tglNodule").onchange=updateNoduleVis;
$("noduleMode").onchange=updateNoduleMode;
$("tglLuna").onchange=updateLunaVis;
// current splash state, respecting toggles (a toggled-off field never counts as a value)
function currentFields(){
  const lunaOn=$("tglLuna").checked;
  const images=$("cfgImages").value.trim();
  const dataset=$("cfgDataset").value.trim(), series=$("cfgSeries").value.trim();
  const masks=$("tglLung").checked?$("cfgMasks").value.trim():"";
  let masks2="",noduleCsv="",noduleCols="";
  if($("tglNodule").checked){
    if($("noduleMode").value==="csv"){noduleCsv=$("cfgNoduleCsv").value.trim();noduleCols=$("cfgNoduleCols").value.trim();}
    else{masks2=$("cfgMasks2").value.trim();}
  }
  return {lunaOn,images,masks,masks2,noduleCsv,noduleCols,dataset,series};
}
async function browseInto(fieldId,kind){
  setSplashErr("");
  try{
    const r=await (await fetch("/api/browse",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({initial:$(fieldId).value.trim(),kind:kind||"folder"})})).json();
    if(r.error){setSplashErr(r.error);return;}
    if(r.path)$(fieldId).value=r.path;
  }catch(e){setSplashErr("browse request failed");}
}
$("browseImages").onclick=()=>browseInto("cfgImages");
$("browseMasks").onclick=()=>browseInto("cfgMasks");
$("browseMasks2").onclick=()=>browseInto("cfgMasks2");
$("browseNoduleCsv").onclick=()=>browseInto("cfgNoduleCsv","file");
$("browseDataset").onclick=()=>browseInto("cfgDataset");
$("browseSeries").onclick=()=>browseInto("cfgSeries","file");
async function startClicked(){
  const f=currentFields();
  const lunaChanged=f.dataset!==orig.dataset||f.series!==orig.series;
  const genericChanged=f.images!==orig.images||f.masks!==orig.masks||f.masks2!==orig.masks2||
    f.noduleCsv!==orig.noduleCsv||f.noduleCols!==orig.noduleCols;
  let body=null;
  if(f.lunaOn&&lunaChanged) body={dataset:f.dataset,series:f.series};
  else if(!f.lunaOn&&genericChanged&&f.images)
    body={images:f.images,masks:f.masks,masks2:f.masks2,nodule_csv:f.noduleCsv,nodule_cols:f.noduleCols};
  if(body){ // something was (re)picked -> reconfigure the backend before entering
    $("startBtn").disabled=true; setSplashErr("loading…");
    try{
      const r=await (await fetch("/api/configure",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body)})).json();
      if(r.error){setSplashErr(r.error);$("startBtn").disabled=false;return;}
      await initConfig();
    }catch(e){setSplashErr("configure request failed");$("startBtn").disabled=false;return;}
    $("startBtn").disabled=false;
  }
  setSplashErr("");dismissSplash();
}
$("startBtn").onclick=startClicked;
document.addEventListener("keydown",e=>{
  if(e.key==="Enter"&&!$("splash").classList.contains("hidden")){e.preventDefault();startClicked();}
});
initConfig();
