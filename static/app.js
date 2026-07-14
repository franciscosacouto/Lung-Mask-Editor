let W=0,H=0,ZOOM=2,tool="paint",brush=14,opacity=.45,showNod=true,showLung=true,layer="lung";
let col0=[255,85,85],col1=[0,230,90],nLayers=2;   // set from /api/config
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
let drawing=false,addMode=true;
function evtXY(e){const rc=mk.getBoundingClientRect();
  return [(e.clientX-rc.left)/ZOOM,(e.clientY-rc.top)/ZOOM];}
mk_events();
function mk_events(){
 ov.addEventListener("contextmenu",e=>e.preventDefault());
 ov.addEventListener("mousedown",e=>{e.preventDefault();pushUndo();drawing=true;
   addMode=!(e.button===2||tool==="erase");const[x,y]=evtXY(e);paintAt(x,y,addMode);});
 window.addEventListener("mousemove",e=>{if(!drawing)return;const[x,y]=evtXY(e);paintAt(x,y,addMode);});
 window.addEventListener("mouseup",()=>drawing=false);
}
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
function setLayer(l){layer=l;
  $("lyLung").classList.toggle("sec",l!=="lung");$("lyNod").classList.toggle("sec",l!=="nodule");
  if(l==="nodule"&&!showNod){showNod=true;$("nod").checked=true;renderNodule();}
  if(l==="lung"&&!showLung){showLung=true;$("lungvis").checked=true;renderMask();}
  setStatus("editing "+l+" layer",l==="nodule"?"#00e65a":"#ff8a8a");}
// auto-save the current edits before moving to another image
async function gotoPid(i){if(i<0||i>=pids.length||i===cur)return;
  if(dirty)await save(); cur=i; await loadSlice(pids[cur]);}
// controls
$("pid").onchange=e=>gotoPid(pids.indexOf(e.target.value));
$("prev").onclick=()=>gotoPid(cur-1);
$("next").onclick=()=>gotoPid(cur+1);
$("nextUn").onclick=nextUnedited;
$("paint").onclick=()=>{tool="paint";$("paint").classList.remove("sec");$("erase").classList.add("sec");};
$("erase").onclick=()=>{tool="erase";$("erase").classList.remove("sec");$("paint").classList.add("sec");};
$("brush").oninput=e=>{brush=+e.target.value;$("bs").textContent=brush;};
$("op").oninput=e=>{opacity=e.target.value/100;renderMask();renderNodule();};
$("zoom").oninput=e=>{ZOOM=e.target.value/10;$("zl").textContent=ZOOM.toFixed(1);applyZoom();};
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
$("save").onclick=save;
$("reset").onclick=async()=>{const r=await (await fetch("/api/slice/"+pids[cur]+"?orig=1")).json();
  pushUndo();pngToMask(r.mask,m=>{mask=m;renderMask();});pngToMask(r.nodule,m=>{nodule=m;renderNodule();});
  setStatus("reset (not yet saved)");};
document.addEventListener("keydown",e=>{
  if(e.ctrlKey&&e.key==="z"){e.preventDefault();$("undo").click();return;}
  if(e.ctrlKey&&e.key==="s"){e.preventDefault();save();return;}
  if(e.key==="1")setLayer("lung"); if(e.key==="2")setLayer("nodule");
  if(e.key==="u")nextUnedited();
  if(e.key==="p")$("paint").click(); if(e.key==="e")$("erase").click();
  if(e.key==="[")$("brush").value=Math.max(1,brush-2),$("brush").oninput({target:$("brush")});
  if(e.key==="]")$("brush").value=Math.min(60,brush+2),$("brush").oninput({target:$("brush")});
  if(e.key==="ArrowLeft")$("prev").click(); if(e.key==="ArrowRight")$("next").click();
});
// apply layer names/colours from the backend, then load the id list
async function initConfig(){
  const c=await (await fetch("/api/config")).json();
  nLayers=c.nlayers; const L=c.layers, rgb=a=>`rgb(${a[0]},${a[1]},${a[2]})`;
  $("title").textContent=c.title||"Mask editor";
  col0=L[0].color;
  $("lyLung").textContent="● "+L[0].name; $("lyLung").style.color=rgb(col0);
  $("sw0t").textContent="Show "+L[0].name; $("dot0").style.background=rgb(col0); $("leg0t").textContent=L[0].name;
  if(nLayers>1){col1=L[1].color;
    $("lyNod").textContent="● "+L[1].name; $("lyNod").style.color=rgb(col1);
    $("sw1t").textContent="Show "+L[1].name; $("dot1").style.background=rgb(col1); $("leg1t").textContent=L[1].name;
  }else{ // single layer: hide the second layer's controls
    for(const id of ["lyNod","sw1","leg1"]) $(id).style.display="none";
  }
  await loadPids();
}
// splash / landing page: red spotlight follows the mouse, dismiss on Start or Enter
$("splash").addEventListener("mousemove",e=>{
  const s=$("splash").style; s.setProperty("--mx",e.clientX+"px"); s.setProperty("--my",e.clientY+"px");});
function dismissSplash(){$("splash").classList.add("hidden");}
$("startBtn").onclick=dismissSplash;
document.addEventListener("keydown",e=>{
  if(e.key==="Enter"&&!$("splash").classList.contains("hidden")){e.preventDefault();dismissSplash();}
});
initConfig();
