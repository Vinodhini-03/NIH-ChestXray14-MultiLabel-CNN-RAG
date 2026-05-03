import { useState, useRef, useEffect, useCallback } from "react";

const API = "https://nih-chestxray14-multilabel-cnn-rag-production.up.railway.app";

const LABELS = [
  "Atelectasis","Cardiomegaly","Effusion","Infiltration",
  "Mass","Nodule","Pneumonia","Pneumothorax",
  "Consolidation","Edema","Emphysema","Fibrosis",
  "Pleural_Thickening","Hernia"
];

const pillStyle = (p) => {
  if (p >= 0.8) return { bg:"rgba(34,197,94,0.12)", border:"rgba(34,197,94,0.35)", color:"#4ade80" };
  if (p >= 0.5) return { bg:"rgba(251,146,60,0.12)", border:"rgba(251,146,60,0.35)", color:"#fb923c" };
  return { bg:"rgba(148,163,184,0.08)", border:"rgba(148,163,184,0.2)", color:"#64748b" };
};

function parseMd(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br/>");
}

// ── Cursor ────────────────────────────────────────────────────────────────────
function Cursor() {
  return <span style={{
    display:"inline-block", width:2, height:"1em",
    background:"#60a5fa", marginLeft:2,
    animation:"blink 1s step-end infinite", verticalAlign:"text-bottom"
  }}/>;
}

// ── Prediction Bars ───────────────────────────────────────────────────────────
function PredictionBar({ label, prob }) {
  const s = pillStyle(prob);
  const pct = Math.round(prob * 100);
  return (
    <div style={{ marginBottom:8 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
        <span style={{ fontSize:"0.8rem", color: prob>=0.5 ? s.color : "#475569", fontWeight: prob>=0.5 ? 600 : 400 }}>
          {label}
        </span>
        <span style={{ fontSize:"0.78rem", color:s.color, fontWeight:700 }}>{pct}%</span>
      </div>
      <div style={{ height:5, borderRadius:4, background:"rgba(255,255,255,0.05)", overflow:"hidden" }}>
        <div style={{
          height:"100%", width:`${pct}%`, borderRadius:4,
          background: prob>=0.8 ? "linear-gradient(90deg,#16a34a,#4ade80)"
                    : prob>=0.5 ? "linear-gradient(90deg,#c2410c,#fb923c)"
                    : "rgba(148,163,184,0.2)",
          transition:"width 1.2s cubic-bezier(.4,0,.2,1)",
        }}/>
      </div>
    </div>
  );
}

// ── Pills ─────────────────────────────────────────────────────────────────────
function Pills({ probs }) {
  const detected = LABELS.map((l,i)=>({l,p:probs[i]})).filter(x=>x.p>=0.5).sort((a,b)=>b.p-a.p);
  const rest = LABELS.map((l,i)=>({l,p:probs[i]})).filter(x=>x.p<0.5).sort((a,b)=>b.p-a.p).slice(0,4);
  return (
    <div style={{ marginBottom:14 }}>
      {detected.length > 0 && (
        <div style={{ marginBottom:6 }}>
          <span style={{ fontSize:"0.68rem", color:"#475569", fontWeight:700, textTransform:"uppercase", letterSpacing:"0.5px" }}>Detected</span>
          <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginTop:5 }}>
            {detected.map(({l,p})=>{
              const s = pillStyle(p);
              return <span key={l} style={{
                padding:"4px 11px", borderRadius:100,
                background:s.bg, border:`1px solid ${s.border}`,
                color:s.color, fontSize:"0.73rem", fontWeight:700,
              }}>{l} · {Math.round(p*100)}%</span>;
            })}
          </div>
        </div>
      )}
      {detected.length === 0 && (
        <span style={{
          padding:"4px 12px", borderRadius:100,
          background:"rgba(34,197,94,0.1)", border:"1px solid rgba(34,197,94,0.25)",
          color:"#4ade80", fontSize:"0.78rem", fontWeight:700,
        }}>No abnormalities detected above threshold</span>
      )}
    </div>
  );
}

// ── Grad-CAM Panel ────────────────────────────────────────────────────────────
function GradCamPanel({ gradcam, topLabel, probs }) {
  const [tab, setTab] = useState("cam");
  return (
    <div style={{
      background:"#080d14", border:"1px solid #1a2535",
      borderRadius:14, overflow:"hidden", marginBottom:18,
    }}>
      <div style={{ display:"flex", borderBottom:"1px solid #1a2535", background:"#0a1020" }}>
        {[["cam","Grad-CAM Heatmap"],["bars","All 14 Scores"]].map(([id,label])=>(
          <button key={id} onClick={()=>setTab(id)} style={{
            padding:"10px 20px", border:"none", cursor:"pointer",
            background:tab===id?"rgba(59,130,246,0.08)":"transparent",
            color:tab===id?"#60a5fa":"#475569",
            borderBottom:tab===id?"2px solid #3b82f6":"2px solid transparent",
            fontSize:"0.78rem", fontWeight:700, letterSpacing:"0.3px",
            transition:"all .2s",
          }}>{label}</button>
        ))}
      </div>
      <div style={{ padding:18 }}>
        {tab==="cam" && (
          <div>
            <div style={{ borderRadius:10, overflow:"hidden", border:"1px solid #1a2535", marginBottom:12 }}>
              <img src={`data:image/png;base64,${gradcam}`} style={{ width:"100%", display:"block" }} alt="Grad-CAM Heatmap"/>
            </div>
            <div style={{
              background:"rgba(59,130,246,0.05)", border:"1px solid rgba(59,130,246,0.12)",
              borderRadius:8, padding:"9px 13px",
              fontSize:"0.73rem", color:"#64748b", lineHeight:1.6,
            }}>
              Activation map for <strong style={{color:"#60a5fa"}}>{topLabel}</strong> — red/yellow regions indicate areas the model weighted most heavily for this prediction.
            </div>
          </div>
        )}
        {tab==="bars" && (
          <div>
            {LABELS.map((l,i)=><PredictionBar key={l} label={l} prob={probs[i]}/>)}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Message ───────────────────────────────────────────────────────────────────
function Message({ msg, isStreaming }) {
  if (msg.role === "user") {
    return (
      <div style={{ display:"flex", justifyContent:"flex-end", marginBottom:22 }}>
        <div style={{ maxWidth:"68%" }}>
          {msg.image && (
            <div style={{
              borderRadius:14, overflow:"hidden",
              border:"1px solid rgba(59,130,246,0.2)",
              marginBottom: msg.content ? 8 : 0,
              boxShadow:"0 4px 20px rgba(0,0,0,0.4)",
            }}>
              <img src={msg.image} alt="X-ray" style={{ width:"100%", display:"block", maxWidth:280 }}/>
            </div>
          )}
          {msg.content && (
            <div style={{
              background:"linear-gradient(135deg,rgba(29,78,216,0.2),rgba(8,145,178,0.15))",
              border:"1px solid rgba(59,130,246,0.2)",
              borderRadius:"16px 16px 4px 16px",
              padding:"12px 16px",
              color:"#e2e8f0", fontSize:"0.88rem", lineHeight:1.65,
            }}>{msg.content}</div>
          )}
          <div style={{ textAlign:"right", marginTop:4, fontSize:"0.68rem", color:"#1e293b" }}>You</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display:"flex", gap:14, marginBottom:28, alignItems:"flex-start" }}>
      {/* Avatar */}
      <div style={{
        width:36, height:36, minWidth:36, borderRadius:10,
        background:"linear-gradient(135deg,#1d4ed8,#0891b2)",
        display:"flex", alignItems:"center", justifyContent:"center",
        fontSize:"0.72rem", fontWeight:800, color:"#fff",
        letterSpacing:"0.3px", marginTop:2, flexShrink:0,
        boxShadow:"0 0 0 3px rgba(59,130,246,0.12)",
      }}>AI</div>

      <div style={{ flex:1, minWidth:0 }}>
        <div style={{
          fontSize:"0.72rem", color:"#3b82f6", fontWeight:800,
          letterSpacing:"1px", textTransform:"uppercase", marginBottom:10,
        }}>ChestAI</div>

        {msg.probs && <Pills probs={msg.probs}/>}
        {msg.gradcam && <GradCamPanel gradcam={msg.gradcam} topLabel={msg.topLabel} probs={msg.probs}/>}

        <div style={{ color:"#cbd5e1", fontSize:"0.88rem", lineHeight:1.78 }}
          dangerouslySetInnerHTML={{ __html: parseMd(msg.content) }}/>

        {isStreaming && <Cursor/>}

        {!isStreaming && msg.content && (
          <div style={{
            marginTop:14,
            background:"rgba(239,68,68,0.05)",
            border:"1px solid rgba(239,68,68,0.15)",
            borderLeft:"3px solid rgba(239,68,68,0.6)",
            borderRadius:"0 8px 8px 0",
            padding:"7px 12px",
            fontSize:"0.72rem", color:"#f87171", lineHeight:1.6,
          }}>
            AI-assisted only — not a diagnostic tool. Consult a qualified radiologist.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Typing Indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div style={{ display:"flex", gap:14, marginBottom:24, alignItems:"flex-start" }}>
      <div style={{
        width:36, height:36, minWidth:36, borderRadius:10,
        background:"linear-gradient(135deg,#1d4ed8,#0891b2)",
        display:"flex", alignItems:"center", justifyContent:"center",
        fontSize:"0.72rem", fontWeight:800, color:"#fff",
      }}>AI</div>
      <div style={{ paddingTop:10 }}>
        <div style={{ display:"flex", gap:5 }}>
          {[0,1,2].map(i=>(
            <div key={i} style={{
              width:7, height:7, borderRadius:"50%", background:"#3b82f6",
              animation:`bounce 1.2s ease-in-out ${i*0.2}s infinite`,
            }}/>
          ))}
        </div>
        <div style={{ fontSize:"0.72rem", color:"#334155", marginTop:5 }}>Analysing X-ray…</div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// HOME PAGE
// ══════════════════════════════════════════════════════════════════════════════
function HomePage({ onStart }) {
  const diseases = [
    {icon:"🫀",name:"Cardiomegaly",auroc:"0.897",desc:"Enlarged heart, CTR > 0.5"},
    {icon:"🌬️",name:"Emphysema",auroc:"0.903",desc:"Permanent airspace enlargement"},
    {icon:"💧",name:"Effusion",auroc:"0.867",desc:"Fluid in pleural space"},
    {icon:"🫁",name:"Pneumothorax",auroc:"0.857",desc:"Air in pleural space"},
    {icon:"💦",name:"Edema",auroc:"0.875",desc:"Fluid in lung interstitium"},
    {icon:"🌀",name:"Atelectasis",auroc:"0.787",desc:"Lung tissue collapse"},
    {icon:"🔴",name:"Mass",auroc:"0.807",desc:"Pulmonary opacity > 3cm"},
    {icon:"⚪",name:"Nodule",auroc:"0.726",desc:"Rounded opacity < 3cm"},
    {icon:"🦠",name:"Pneumonia",auroc:"0.719",desc:"Air sac infection"},
    {icon:"🌫️",name:"Consolidation",auroc:"0.804",desc:"Alveolar fluid/cells"},
    {icon:"🧵",name:"Fibrosis",auroc:"0.801",desc:"Lung scarring"},
    {icon:"📍",name:"Infiltration",auroc:"0.697",desc:"Dense lung parenchyma"},
    {icon:"📏",name:"Pl. Thickening",auroc:"0.796",desc:"Pleural scarring"},
    {icon:"🔺",name:"Hernia",auroc:"0.914",desc:"Abdominal contents in chest"},
  ];

  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior:"smooth" });

  return (
    <div style={{ minHeight:"100vh", background:"#060910", color:"#e2e8f0" }}>

      {/* Grid bg */}
      <div style={{
        position:"fixed", inset:0, pointerEvents:"none", zIndex:0,
        backgroundImage:"linear-gradient(rgba(59,130,246,0.025) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,0.025) 1px,transparent 1px)",
        backgroundSize:"60px 60px",
      }}/>
      <div style={{ position:"fixed", top:-200, right:-100, width:600, height:600,
        background:"radial-gradient(circle,rgba(29,78,216,0.07) 0%,transparent 70%)", pointerEvents:"none", zIndex:0 }}/>
      <div style={{ position:"fixed", bottom:-200, left:-100, width:500, height:500,
        background:"radial-gradient(circle,rgba(8,145,178,0.05) 0%,transparent 70%)", pointerEvents:"none", zIndex:0 }}/>

      <div style={{ position:"relative", zIndex:1 }}>

        {/* Navbar */}
        <nav style={{
          display:"flex", alignItems:"center", justifyContent:"space-between",
          padding:"1rem 3rem",
          borderBottom:"1px solid rgba(30,36,50,0.8)",
          backdropFilter:"blur(12px)",
          background:"rgba(6,9,16,0.9)",
          position:"sticky", top:0, zIndex:100,
        }}>
          <div style={{ fontSize:"1.3rem", fontWeight:900, letterSpacing:"-0.5px", color:"#fff", fontFamily:"'DM Sans',sans-serif" }}>
            Chest<span style={{ color:"#3b82f6" }}>AI</span>
          </div>
          <div style={{ display:"flex", gap:"2.5rem", alignItems:"center" }}>
            {[["Diseases","diseases"],["How it Works","howitworks"],["Models","models"]].map(([l,id])=>(
              <span key={l} onClick={()=>scrollTo(id)} style={{
                color:"#64748b", fontSize:"0.85rem", cursor:"pointer", transition:"color .2s",
              }}
                onMouseEnter={e=>e.target.style.color="#e2e8f0"}
                onMouseLeave={e=>e.target.style.color="#64748b"}
              >{l}</span>
            ))}
            <button onClick={onStart} style={{
              padding:"9px 22px",
              background:"linear-gradient(135deg,#1d4ed8,#0891b2)",
              border:"none", borderRadius:10, color:"#fff",
              fontSize:"0.85rem", fontWeight:700, cursor:"pointer",
              boxShadow:"0 0 20px rgba(59,130,246,0.3)", transition:"all .2s",
            }}
              onMouseEnter={e=>{ e.target.style.transform="scale(1.04)"; e.target.style.boxShadow="0 0 30px rgba(59,130,246,0.5)"; }}
              onMouseLeave={e=>{ e.target.style.transform="scale(1)"; e.target.style.boxShadow="0 0 20px rgba(59,130,246,0.3)"; }}
            >Analyse X-ray →</button>
          </div>
        </nav>

        {/* Hero */}
        <div style={{ display:"flex", flexDirection:"column", alignItems:"center", textAlign:"center", padding:"6rem 2rem 4rem" }}>
          <div style={{
            display:"inline-flex", alignItems:"center", gap:8,
            background:"rgba(59,130,246,0.07)", border:"1px solid rgba(59,130,246,0.18)",
            borderRadius:100, padding:"5px 18px",
            fontSize:"0.72rem", fontWeight:700, color:"#60a5fa",
            letterSpacing:"1.5px", textTransform:"uppercase", marginBottom:"1.8rem",
          }}>
            <span style={{ width:6, height:6, borderRadius:"50%", background:"#22c55e",
              animation:"pulse 2s infinite", display:"inline-block" }}/>
            NIH ChestX-ray14 · ResNet-18 · RAG + LLM
          </div>

          <h1 style={{
            fontSize:"clamp(2.8rem,6vw,4.8rem)", fontWeight:900, lineHeight:1.05,
            letterSpacing:"-2px", color:"#fff", marginBottom:"1.4rem",
            fontFamily:"'DM Sans',sans-serif",
          }}>
            AI-Powered<br/>
            <span style={{
              background:"linear-gradient(135deg,#3b82f6 0%,#06b6d4 50%,#0891b2 100%)",
              WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent",
            }}>Chest X-ray Analysis</span>
          </h1>

          <p style={{ fontSize:"1.05rem", color:"#64748b", maxWidth:560, lineHeight:1.75, marginBottom:"2.5rem" }}>
            Upload a chest X-ray and get AI-powered detection of <strong style={{color:"#94a3b8"}}>14 thoracic conditions</strong> — with Grad-CAM explainability, RAG-grounded knowledge, and LLM interpretation you can ask follow-up questions to.
          </p>

          <button onClick={onStart} style={{
            padding:"15px 38px",
            background:"linear-gradient(135deg,#1d4ed8,#0891b2)",
            border:"none", borderRadius:12, color:"#fff",
            fontSize:"1.05rem", fontWeight:700, cursor:"pointer",
            boxShadow:"0 8px 40px rgba(59,130,246,0.35)",
            transition:"all .2s", marginBottom:"3.5rem",
            display:"flex", alignItems:"center", gap:10,
          }}
            onMouseEnter={e=>{ e.currentTarget.style.transform="translateY(-2px)"; e.currentTarget.style.boxShadow="0 14px 50px rgba(59,130,246,0.45)"; }}
            onMouseLeave={e=>{ e.currentTarget.style.transform="translateY(0)"; e.currentTarget.style.boxShadow="0 8px 40px rgba(59,130,246,0.35)"; }}
          >
            Start Analysis →
          </button>

          {/* Stats */}
          <div style={{
            display:"flex", gap:"3rem",
            padding:"1.8rem 3.5rem",
            background:"rgba(13,17,23,0.8)",
            border:"1px solid rgba(30,41,59,0.8)",
            borderRadius:18, backdropFilter:"blur(12px)",
          }}>
            {[["112K+","Training Images"],["14","Disease Labels"],["0.8179","Mean AUROC"],["ResNet-18","Best Model"]].map(([v,l])=>(
              <div key={l} style={{ textAlign:"center" }}>
                <div style={{ fontSize:"1.7rem", fontWeight:900, color:"#fff",
                  letterSpacing:"-1px", fontFamily:"'DM Sans',sans-serif" }}>{v}</div>
                <div style={{ fontSize:"0.72rem", color:"#475569", marginTop:4, letterSpacing:"0.3px" }}>{l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Disclaimer */}
        <div style={{ maxWidth:680, margin:"0 auto", padding:"0 2rem 3rem" }}>
          <div style={{
            background:"rgba(239,68,68,0.05)", border:"1px solid rgba(239,68,68,0.18)",
            borderLeft:"3px solid rgba(239,68,68,0.5)",
            borderRadius:"0 10px 10px 0", padding:"10px 16px",
            fontSize:"0.8rem", color:"#fca5a5", lineHeight:1.6, textAlign:"center",
          }}>
            ⚠️ <strong>Not a diagnostic tool.</strong> All outputs must be reviewed by a qualified radiologist before any clinical decision.
          </div>
        </div>

        <div style={{ borderTop:"1px solid rgba(30,41,59,0.5)", margin:"0 3rem" }}/>

        {/* Diseases */}
        <div id="diseases" style={{ maxWidth:1100, margin:"0 auto", padding:"4rem 2rem" }}>
          <div style={{ textAlign:"center", marginBottom:"2.5rem" }}>
            <h2 style={{ fontSize:"2rem", fontWeight:900, color:"#fff",
              letterSpacing:"-0.5px", fontFamily:"'DM Sans',sans-serif" }}>14 Detectable Conditions</h2>
            <p style={{ color:"#475569", fontSize:"0.85rem", marginTop:8 }}>
              Trained with patient-wise splits and class-weighted BCE loss to handle severe imbalance
            </p>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(7,1fr)", gap:10 }}>
            {diseases.map(d=>(
              <div key={d.name} style={{
                background:"#0d1117", border:"1px solid rgba(30,41,59,0.8)",
                borderRadius:12, padding:"1.1rem 0.8rem", textAlign:"center",
                transition:"all .2s", cursor:"default",
              }}
                onMouseEnter={e=>{ e.currentTarget.style.borderColor="rgba(59,130,246,0.4)"; e.currentTarget.style.transform="translateY(-3px)"; e.currentTarget.style.background="#0f1923"; }}
                onMouseLeave={e=>{ e.currentTarget.style.borderColor="rgba(30,41,59,0.8)"; e.currentTarget.style.transform="translateY(0)"; e.currentTarget.style.background="#0d1117"; }}
              >
                <div style={{ fontSize:"1.5rem", marginBottom:6 }}>{d.icon}</div>
                <div style={{ fontSize:"0.72rem", fontWeight:700, color:"#cbd5e1", marginBottom:5, lineHeight:1.3 }}>{d.name}</div>
                <div style={{ fontSize:"0.65rem", color:"#334155", marginBottom:6, lineHeight:1.4 }}>{d.desc}</div>
                <div style={{
                  display:"inline-block",
                  background:"rgba(59,130,246,0.08)", border:"1px solid rgba(59,130,246,0.18)",
                  color:"#60a5fa", padding:"2px 8px", borderRadius:5,
                  fontSize:"0.65rem", fontWeight:700,
                }}>{d.auroc}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderTop:"1px solid rgba(30,41,59,0.5)", margin:"0 3rem" }}/>

        {/* How it works */}
        <div id="howitworks" style={{ maxWidth:960, margin:"0 auto", padding:"4rem 2rem" }}>
          <div style={{ textAlign:"center", marginBottom:"2.5rem" }}>
            <h2 style={{ fontSize:"2rem", fontWeight:900, color:"#fff",
              letterSpacing:"-0.5px", fontFamily:"'DM Sans',sans-serif" }}>How It Works</h2>
            <p style={{ color:"#475569", fontSize:"0.85rem", marginTop:8 }}>
              Three-stage pipeline from raw image to grounded LLM interpretation
            </p>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:18 }}>
            {[
              ["01","Upload & Predict","Drag your X-ray into the chat. ResNet-18 runs inference and outputs probability scores for all 14 disease labels simultaneously.","#3b82f6"],
              ["02","RAG Retrieval","Detected findings query a curated medical knowledge base via FAISS vector search. Top 3 relevant chunks are retrieved as citations.","#06b6d4"],
              ["03","LLM Interpretation","Groq LLaMA-3.3-70B generates a structured, citation-backed analysis. Then keep chatting — ask any follow-up question about the findings.","#0891b2"],
            ].map(([n,t,d,c])=>(
              <div key={n} style={{
                background:"#0d1117", border:"1px solid rgba(30,41,59,0.8)",
                borderRadius:16, padding:"1.8rem",
                transition:"all .2s",
              }}
                onMouseEnter={e=>{ e.currentTarget.style.borderColor=`${c}40`; e.currentTarget.style.transform="translateY(-3px)"; }}
                onMouseLeave={e=>{ e.currentTarget.style.borderColor="rgba(30,41,59,0.8)"; e.currentTarget.style.transform="translateY(0)"; }}
              >
                <div style={{
                  width:38, height:38, borderRadius:10,
                  background:`${c}18`, border:`1px solid ${c}30`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:"0.82rem", fontWeight:900, color:c, marginBottom:"1.1rem",
                }}>{n}</div>
                <div style={{ fontSize:"1rem", fontWeight:700, color:"#e2e8f0", marginBottom:8 }}>{t}</div>
                <div style={{ fontSize:"0.8rem", color:"#475569", lineHeight:1.7 }}>{d}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ borderTop:"1px solid rgba(30,41,59,0.5)", margin:"0 3rem" }}/>

        {/* Models */}
        <div id="models" style={{ maxWidth:960, margin:"0 auto", padding:"4rem 2rem" }}>
          <div style={{ textAlign:"center", marginBottom:"2.5rem" }}>
            <h2 style={{ fontSize:"2rem", fontWeight:900, color:"#fff",
              letterSpacing:"-0.5px", fontFamily:"'DM Sans',sans-serif" }}>Model Comparison</h2>
            <p style={{ color:"#475569", fontSize:"0.85rem", marginTop:8 }}>
              Three CNN architectures trained and evaluated on NIH ChestX-ray14
            </p>
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:18 }}>
            {[
              { name:"ResNet-18", auroc:"0.8179", status:"Best Model", color:"#22c55e", desc:"Residual connections enable deeper training without vanishing gradients. Best generalisation across all 14 labels." },
              { name:"VGG-19", auroc:"0.7934", status:"Baseline", color:"#3b82f6", desc:"Deep sequential conv layers with large receptive field. Strong feature extractor but computationally heavier." },
              { name:"Custom CNN", auroc:"0.7612", status:"Experimental", color:"#8b5cf6", desc:"Purpose-built architecture with batch norm, dropout, and multi-scale feature maps for chest pathology." },
            ].map(m=>(
              <div key={m.name} style={{
                background:"#0d1117", border:`1px solid ${m.color}25`,
                borderRadius:16, padding:"1.8rem",
                transition:"all .2s",
              }}
                onMouseEnter={e=>{ e.currentTarget.style.borderColor=`${m.color}50`; e.currentTarget.style.transform="translateY(-3px)"; }}
                onMouseLeave={e=>{ e.currentTarget.style.borderColor=`${m.color}25`; e.currentTarget.style.transform="translateY(0)"; }}
              >
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:12 }}>
                  <div style={{ fontSize:"1.1rem", fontWeight:800, color:"#e2e8f0", fontFamily:"'DM Sans',sans-serif" }}>{m.name}</div>
                  <span style={{
                    padding:"3px 10px", borderRadius:100,
                    background:`${m.color}15`, border:`1px solid ${m.color}30`,
                    color:m.color, fontSize:"0.68rem", fontWeight:700,
                  }}>{m.status}</span>
                </div>
                <div style={{ fontSize:"2rem", fontWeight:900, color:m.color, letterSpacing:"-1px", marginBottom:8 }}>{m.auroc}</div>
                <div style={{ fontSize:"0.68rem", color:"#475569", marginBottom:12 }}>Mean AUROC</div>
                <div style={{ fontSize:"0.78rem", color:"#475569", lineHeight:1.65 }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div style={{
          borderTop:"1px solid rgba(30,41,59,0.5)",
          padding:"2rem 3rem",
          display:"flex", justifyContent:"space-between", alignItems:"center",
          flexWrap:"wrap", gap:12,
        }}>
          <div style={{ fontSize:"0.85rem", fontWeight:800, color:"#334155", fontFamily:"'DM Sans',sans-serif" }}>
            Chest<span style={{ color:"#1d4ed8" }}>AI</span>
          </div>
          <div style={{ fontSize:"0.72rem", color:"#1e293b" }}>
            NIH ChestX-ray14 · ResNet-18 · FAISS · Groq LLaMA-3.3-70B · Not a diagnostic tool.
          </div>
          <a href="https://github.com/Vinodhini-03/NIH-ChestXray14-MultiLabel-CNN-RAG"
            target="_blank" rel="noreferrer"
            style={{
              fontSize:"0.75rem", color:"#334155", textDecoration:"none",
              transition:"color .2s",
            }}
            onMouseEnter={e=>e.target.style.color="#60a5fa"}
            onMouseLeave={e=>e.target.style.color="#334155"}
          >GitHub →</a>
        </div>
      </div>

      {/* Floating CTA */}
      <button onClick={onStart} style={{
        position:"fixed", bottom:"2rem", right:"2rem",
        width:58, height:58, borderRadius:"50%",
        background:"linear-gradient(135deg,#1d4ed8,#0891b2)",
        border:"none", cursor:"pointer", zIndex:200,
        display:"flex", alignItems:"center", justifyContent:"center",
        boxShadow:"0 4px 24px rgba(59,130,246,0.5), 0 0 0 4px rgba(59,130,246,0.12)",
        animation:"floatPulse 2.5s ease-in-out infinite",
      }} title="Start Analysis">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </button>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// CHAT PAGE
// ══════════════════════════════════════════════════════════════════════════════
function ChatPage({ onHome }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [probs, setProbs]       = useState(null);
  const bottomRef = useRef(null);
  const fileRef   = useRef(null);
  const inputRef  = useRef(null);

  useEffect(()=>{
    bottomRef.current?.scrollIntoView({ behavior:"smooth" });
  }, [messages, loading]);

  const analyseImage = useCallback(async (file) => {
    const url = URL.createObjectURL(file);
    setMessages(prev=>[...prev, { role:"user", content:"", image:url }]);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API}/predict`, { method:"POST", body:fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setProbs(data.probs);
      setMessages(prev=>[...prev, {
        role:"assistant", content:data.summary,
        probs:data.probs, gradcam:data.gradcam, topLabel:data.top_label,
      }]);
    } catch(e) {
      setMessages(prev=>[...prev, {
        role:"assistant",
        content:`Connection error — make sure the FastAPI backend is running.\n\nIn your terminal:\n**cd backend**\n**uvicorn main:app --reload --port 8000**`,
      }]);
    } finally { setLoading(false); }
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    setMessages(prev=>[...prev, { role:"user", content:text }]);

    const msgId = Date.now();
    setMessages(prev=>[...prev, { role:"assistant", content:"", _id:msgId, _streaming:true }]);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ question:text, probs: probs || new Array(14).fill(0) }),
      });
      if (!res.ok) throw new Error(res.statusText);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const lines = decoder.decode(value).split("\n").filter(l=>l.startsWith("data:"));
        for (const line of lines) {
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") break;
          try {
            const { text:t } = JSON.parse(payload);
            full += t;
            setMessages(prev=>prev.map(m=>m._id===msgId ? {...m, content:full} : m));
          } catch {}
        }
      }
      setMessages(prev=>prev.map(m=>m._id===msgId ? {...m, _streaming:false} : m));
    } catch(e) {
      setMessages(prev=>prev.map(m=>m._id===msgId
        ? {...m, content:"Connection error — is the backend running on port 8000?", _streaming:false}
        : m));
    }
  }, [input, loading, probs]);

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) analyseImage(file);
  }, [analyseImage]);

  const suggested = [
    "What does this finding mean clinically?",
    "How confident is the model in these results?",
    "What are the limitations of this prediction?",
    "Should I be concerned about these findings?",
  ];

  return (
    <div style={{
      height:"100vh", display:"flex", flexDirection:"column",
      background:"#070b12", color:"#e2e8f0",
    }}
      onDragOver={e=>{ e.preventDefault(); setDragOver(true); }}
      onDragLeave={()=>setDragOver(false)}
      onDrop={onDrop}
    >
      {/* Drag overlay */}
      {dragOver && (
        <div style={{
          position:"fixed", inset:0, zIndex:999,
          background:"rgba(6,9,16,0.88)",
          display:"flex", alignItems:"center", justifyContent:"center",
          border:"2px dashed #3b82f6", backdropFilter:"blur(6px)",
        }}>
          <div style={{ textAlign:"center" }}>
            <div style={{
              width:80, height:80, borderRadius:20, margin:"0 auto 16px",
              background:"rgba(59,130,246,0.12)", border:"1px solid rgba(59,130,246,0.3)",
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:"2.2rem",
            }}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <div style={{ fontSize:"1.3rem", fontWeight:700, color:"#60a5fa" }}>Drop X-ray to Analyse</div>
            <div style={{ color:"#475569", fontSize:"0.85rem", marginTop:6 }}>PNG, JPG, JPEG supported</div>
          </div>
        </div>
      )}

      {/* Top bar */}
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"0.85rem 1.8rem",
        borderBottom:"1px solid rgba(20,30,50,0.9)",
        background:"rgba(7,11,18,0.97)",
        backdropFilter:"blur(12px)",
        flexShrink:0, zIndex:50,
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <button onClick={onHome} style={{
            padding:"7px 16px", border:"1px solid rgba(30,41,59,0.8)",
            borderRadius:8, background:"transparent", color:"#475569",
            fontSize:"0.8rem", cursor:"pointer", transition:"all .2s", fontWeight:600,
          }}
            onMouseEnter={e=>{ e.currentTarget.style.color="#e2e8f0"; e.currentTarget.style.borderColor="rgba(59,130,246,0.4)"; }}
            onMouseLeave={e=>{ e.currentTarget.style.color="#475569"; e.currentTarget.style.borderColor="rgba(30,41,59,0.8)"; }}
          >← Home</button>

          <div style={{ width:1, height:28, background:"rgba(30,41,59,0.8)" }}/>

          <div>
            <div style={{ display:"flex", alignItems:"center", gap:8 }}>
              <span style={{ fontSize:"0.95rem", fontWeight:800, color:"#e2e8f0", fontFamily:"'DM Sans',sans-serif" }}>
                Chest<span style={{ color:"#3b82f6" }}>AI</span>
              </span>
              <span style={{ width:7, height:7, borderRadius:"50%", background:"#22c55e",
                animation:"pulse 2s infinite", display:"inline-block" }}/>
            </div>
            <div style={{ fontSize:"0.68rem", color:"#334155", marginTop:1 }}>
              ResNet-18 · FAISS RAG · Groq LLaMA-3.3-70B
            </div>
          </div>
        </div>

        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          {probs && (
            <div style={{
              padding:"5px 13px",
              background:"rgba(34,197,94,0.08)", border:"1px solid rgba(34,197,94,0.2)",
              borderRadius:8, fontSize:"0.72rem", color:"#4ade80", fontWeight:700,
            }}>✓ X-ray analysed</div>
          )}
          <button onClick={()=>{ setMessages([]); setProbs(null); }} style={{
            padding:"7px 16px", border:"1px solid rgba(30,41,59,0.8)",
            borderRadius:8, background:"transparent", color:"#475569",
            fontSize:"0.8rem", cursor:"pointer", transition:"all .2s", fontWeight:600,
          }}
            onMouseEnter={e=>{ e.currentTarget.style.color="#e2e8f0"; }}
            onMouseLeave={e=>{ e.currentTarget.style.color="#475569"; }}
          >New Chat</button>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex:1, overflowY:"auto", padding:"2rem 1.5rem" }}>
        <div style={{ maxWidth:780, width:"100%", margin:"0 auto" }}>

          {messages.length === 0 && (
            <div style={{
              display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"center",
              minHeight:"65vh", textAlign:"center",
            }}>
              <div style={{
                width:72, height:72, borderRadius:18,
                background:"linear-gradient(135deg,rgba(29,78,216,0.15),rgba(8,145,178,0.1))",
                border:"1px solid rgba(59,130,246,0.2)",
                display:"flex", alignItems:"center", justifyContent:"center",
                marginBottom:"1.5rem",
              }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
              </div>

              <h2 style={{
                fontSize:"1.7rem", fontWeight:900, color:"#e2e8f0",
                letterSpacing:"-0.5px", marginBottom:"0.6rem",
                fontFamily:"'DM Sans',sans-serif",
              }}>Ready to Analyse</h2>
              <p style={{ color:"#334155", fontSize:"0.88rem", lineHeight:1.7, maxWidth:420, marginBottom:"2.5rem" }}>
                Upload a chest X-ray using the <strong style={{
                  background:"rgba(59,130,246,0.1)", border:"1px solid rgba(59,130,246,0.2)",
                  padding:"1px 8px", borderRadius:5, color:"#60a5fa", fontSize:"0.8rem",
                }}>📎</strong> button below, or drag and drop anywhere on this page.
              </p>

              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, maxWidth:500, width:"100%" }}>
                {suggested.map(s=>(
                  <button key={s} onClick={()=>{ setInput(s); inputRef.current?.focus(); }} style={{
                    padding:"12px 16px",
                    background:"#0d1117", border:"1px solid rgba(20,30,50,0.9)",
                    borderRadius:12, textAlign:"left",
                    color:"#475569", fontSize:"0.8rem", lineHeight:1.5,
                    cursor:"pointer", transition:"all .2s",
                  }}
                    onMouseEnter={e=>{ e.currentTarget.style.borderColor="rgba(59,130,246,0.3)"; e.currentTarget.style.color="#94a3b8"; e.currentTarget.style.background="#0f1520"; }}
                    onMouseLeave={e=>{ e.currentTarget.style.borderColor="rgba(20,30,50,0.9)"; e.currentTarget.style.color="#475569"; e.currentTarget.style.background="#0d1117"; }}
                  >
                    <div style={{ fontSize:"0.65rem", color:"#1e293b", marginBottom:4, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.5px" }}>Try asking</div>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i)=>(
            <Message key={i} msg={msg} isStreaming={msg._streaming}/>
          ))}
          {loading && <TypingIndicator/>}
          <div ref={bottomRef}/>
        </div>
      </div>

      {/* Input bar */}
      <div style={{
        borderTop:"1px solid rgba(20,30,50,0.9)",
        padding:"1.2rem 1.8rem 1.8rem",
        background:"rgba(7,11,18,0.98)",
        backdropFilter:"blur(12px)",
        flexShrink:0,
      }}>
        <div style={{ maxWidth:780, margin:"0 auto" }}>
          <div style={{
            display:"flex", alignItems:"flex-end", gap:10,
            background:"#0d1117",
            border:`1px solid ${dragOver?"#3b82f6":"rgba(20,30,50,0.95)"}`,
            borderRadius:16, padding:"12px 14px",
            transition:"border-color .2s, box-shadow .2s",
            boxShadow:"0 0 0 0 rgba(59,130,246,0)",
          }}
            onFocusCapture={e=>{ e.currentTarget.style.borderColor="rgba(59,130,246,0.4)"; e.currentTarget.style.boxShadow="0 0 0 3px rgba(59,130,246,0.06)"; }}
            onBlurCapture={e=>{ e.currentTarget.style.borderColor="rgba(20,30,50,0.95)"; e.currentTarget.style.boxShadow="none"; }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display:"none" }} onChange={e=>{ const f=e.target.files[0]; if(f) analyseImage(f); e.target.value=""; }}/>
            <button onClick={()=>fileRef.current?.click()} title="Upload X-ray" style={{
              width:38, height:38, borderRadius:10, border:"none",
              background:"rgba(59,130,246,0.08)", color:"#3b82f6",
              fontSize:"1rem", cursor:"pointer",
              display:"flex", alignItems:"center", justifyContent:"center",
              transition:"all .2s", flexShrink:0,
            }}
              onMouseEnter={e=>e.currentTarget.style.background="rgba(59,130,246,0.18)"}
              onMouseLeave={e=>e.currentTarget.style.background="rgba(59,130,246,0.08)"}
            >📎</button>

            <textarea
              ref={inputRef}
              value={input}
              onChange={e=>setInput(e.target.value)}
              onKeyDown={e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); sendMessage(); } }}
              placeholder="Ask about the X-ray findings, disease meanings, confidence levels…"
              rows={1}
              style={{
                flex:1, background:"transparent", border:"none", outline:"none",
                color:"#e2e8f0", fontSize:"0.9rem", lineHeight:1.6,
                resize:"none", fontFamily:"inherit", maxHeight:130, overflowY:"auto",
              }}
            />

            <button onClick={sendMessage} disabled={!input.trim()||loading} style={{
              width:38, height:38, borderRadius:10, border:"none",
              background: input.trim()&&!loading
                ? "linear-gradient(135deg,#1d4ed8,#0891b2)"
                : "rgba(20,30,50,0.8)",
              color: input.trim()&&!loading ? "#fff" : "#1e293b",
              cursor: input.trim()&&!loading ? "pointer" : "not-allowed",
              display:"flex", alignItems:"center", justifyContent:"center",
              transition:"all .2s", flexShrink:0, fontSize:"1rem", fontWeight:700,
            }}>↑</button>
          </div>
          <div style={{ textAlign:"center", marginTop:8, fontSize:"0.68rem", color:"#0f172a" }}>
            ChestAI answers questions about chest X-ray findings and thoracic diseases only · Not a diagnostic tool
          </div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// APP ROOT
// ══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState("home");
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;0,9..40,900&display=swap');
        *, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }
        html { scroll-behavior:smooth; }
        body { background:#060910; font-family:'DM Sans',sans-serif; -webkit-font-smoothing:antialiased; }
        ::-webkit-scrollbar { width:5px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:#1e293b; border-radius:3px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-8px)} }
        @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(34,197,94,0.5)} 70%{box-shadow:0 0 0 8px rgba(34,197,94,0)} 100%{box-shadow:0 0 0 0 rgba(34,197,94,0)} }
        @keyframes floatPulse { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
      `}</style>
      {page==="home" ? <HomePage onStart={()=>setPage("chat")}/> : <ChatPage onHome={()=>setPage("home")}/>}
    </>
  );
}