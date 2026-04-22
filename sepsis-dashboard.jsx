import { useState, useEffect, useRef } from "react";
import datasetInfo from "./data/dataset_audit_summary.json";
import {
  LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, PieChart, Pie, ScatterChart, Scatter
} from "recharts";

// Ini theme colours nya
const C = {
  bg: "#050d1a",
  surface: "#0a1628",
  card: "#0f1f38",
  cardBright: "#142848",
  border: "#1e3a5f",
  borderGlow: "#2563eb",
  text: "#e2eaf8",
  muted: "#6b8ab0",
  accent: "#38bdf8",
  accentGlow: "#0ea5e9",
  green: "#34d399",
  red: "#f87171",
  yellow: "#fbbf24",
  purple: "#a78bfa",
  teal: "#2dd4bf",
};

const fontStack = `'IBM Plex Mono', 'Courier New', monospace`;
const sansStack = `'IBM Plex Sans', 'Segoe UI', sans-serif`;

// Global Stylings
const GlobalStyle = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }
    body { background: ${C.bg}; color: ${C.text}; font-family: ${sansStack}; overflow-x: hidden; }
    ::-webkit-scrollbar { width: 6px; } 
    ::-webkit-scrollbar-track { background: ${C.surface}; }
    ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 3px; }
    .pulse { animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.5 } }
    .fadeIn { animation: fadeIn .4s ease forwards; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:translateY(0) } }
    .glow-border { box-shadow: 0 0 0 1px ${C.borderGlow}, 0 0 20px rgba(37,99,235,.15); }
    .glow-text { text-shadow: 0 0 20px rgba(56,189,248,.6); }
    .scan-line::after {
      content:''; position:absolute; inset:0; background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(56,189,248,.015) 2px,rgba(56,189,248,.015) 4px);
      pointer-events:none; border-radius:inherit;
    }
    input[type=range] { -webkit-appearance:none; appearance:none; height:4px; border-radius:2px; background:${C.border}; outline:none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; border-radius:50%; background:${C.accent}; cursor:pointer; }
    input[type=number], input[type=text], select { background:${C.surface}; border:1px solid ${C.border}; color:${C.text}; border-radius:6px; padding:6px 10px; font-family:${fontStack}; font-size:12px; outline:none; width:100%; }
    input[type=number]:focus, select:focus { border-color:${C.accentGlow}; box-shadow:0 0 0 2px rgba(14,165,233,.2); }
  `}</style>
);

// Template Helpers
const card = (extra = "") => ({
  background: C.card,
  border: `1px solid ${C.border}`,
  borderRadius: 12,
  padding: 20,
  position: "relative",
  overflow: "hidden",
  ...parseExtra(extra),
});
function parseExtra(s) {
  const map = {};
  if (s.includes("glow")) { map.boxShadow = `0 0 0 1px ${C.borderGlow}, 0 0 30px rgba(37,99,235,.15)`; map.borderColor = C.borderGlow; }
  return map;
}

const Tag = ({ color = C.accent, children }) => (
  <span style={{ background: color + "20", color, border: `1px solid ${color}40`, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontFamily: fontStack, fontWeight: 500 }}>
    {children}
  </span>
);

const MetricCard = ({ label, value, unit = "", delta, color = C.accent, icon }) => (
  <div style={{ ...card(), padding: 16 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
      <div>
        <div style={{ fontSize: 11, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{label}</div>
        <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: fontStack, lineHeight: 1 }}>
          {value}<span style={{ fontSize: 14, color: C.muted, marginLeft: 4 }}>{unit}</span>
        </div>
        {delta && <div style={{ fontSize: 11, color: delta > 0 ? C.green : C.red, marginTop: 4, fontFamily: fontStack }}>{delta > 0 ? "▲" : "▼"} {Math.abs(delta)}%</div>}
      </div>
      {icon && <div style={{ fontSize: 28, opacity: .6 }}>{icon}</div>}
    </div>
  </div>
);

const SectionHeader = ({ title, subtitle, badge }) => (
  <div style={{ marginBottom: 28 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: C.text, fontFamily: fontStack }}>{title}</h2>
      {badge && <Tag>{badge}</Tag>}
    </div>
    {subtitle && <p style={{ color: C.muted, fontSize: 13, maxWidth: 560 }}>{subtitle}</p>}
  </div>
);

const RiskGauge = ({ value, label, color }) => {
  const deg = value * 1.8;
  return (
    <div style={{ textAlign: "center", padding: "12px 8px" }}>
      <div style={{ position: "relative", width: 90, height: 50, margin: "0 auto 8px" }}>
        <svg viewBox="0 0 100 55" width="100%" height="100%">
          <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={C.border} strokeWidth="8" strokeLinecap="round" />
          <path d="M 10 50 A 40 40 0 0 1 90 50" fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
            strokeDasharray={`${(deg / 180) * 125.7} 125.7`} />
        </svg>
        <div style={{ position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)", fontSize: 14, fontWeight: 700, color, fontFamily: fontStack }}>
          {value}%
        </div>
      </div>
      <div style={{ fontSize: 11, color: C.muted, fontFamily: fontStack }}>{label}</div>
    </div>
  );
};

// ─── MOCK DATA ────────────────────────────────────────────────────────────────
const ageDistribution = [
  { age: "18-30", count: 45, pct: 9 }, { age: "31-45", count: 120, pct: 24 },
  { age: "46-60", count: 178, pct: 36 }, { age: "61-75", count: 132, pct: 26 },
  { age: "76+", count: 25, pct: 5 },
];
const mortalityByAge = [
  { age: "18-30", mortality: 12 }, { age: "31-45", mortality: 18 },
  { age: "46-60", mortality: 27 }, { age: "61-75", mortality: 42 }, { age: "76+", mortality: 61 },
];
const treatmentFreq = [
  { name: "Antibiotics", count: 487, pct: 97 }, { name: "IV Fluids", count: 451, pct: 90 },
  { name: "Vasopressors", count: 203, pct: 41 }, { name: "Oxygen Therapy", count: 389, pct: 78 },
  { name: "Mechanical Vent", count: 145, pct: 29 },
];
const lossData = Array.from({ length: 50 }, (_, i) => ({
  epoch: i + 1,
  train: +(2.4 * Math.exp(-0.06 * i) + 0.15 + Math.random() * 0.05).toFixed(3),
  val: +(2.6 * Math.exp(-0.055 * i) + 0.22 + Math.random() * 0.06).toFixed(3),
}));
const rocData = Array.from({ length: 20 }, (_, i) => {
  const fpr = i / 19;
  return { fpr: +fpr.toFixed(2), tpr: +(Math.min(1, fpr + (1 - fpr) * 0.87 + (Math.random() - 0.5) * 0.04)).toFixed(3) };
});
const shockProgression = [
  { hour: 0, risk: 18 }, { hour: 6, risk: 22 }, { hour: 12, risk: 31 },
  { hour: 18, risk: 41 }, { hour: 24, risk: 52 }, { hour: 30, risk: 61 }, { hour: 36, risk: 68 },
];

const DEMO_PATIENTS = [
  { name: "Patient A — High Risk", age: 68, hr: 112, map: 58, lactate: 4.2, creatinine: 2.8, wbc: 18.4, temp: 38.9, vasopressor: 1, fluids: 2, antibiotics: 1, sofa: 8, hours: 12 },
  { name: "Patient B — Moderate", age: 52, hr: 96, map: 72, lactate: 2.1, creatinine: 1.4, wbc: 13.2, temp: 38.1, vasopressor: 0, fluids: 1, antibiotics: 1, sofa: 4, hours: 6 },
  { name: "Patient C — Low Risk", age: 38, hr: 84, map: 82, lactate: 1.2, creatinine: 0.9, wbc: 10.8, temp: 37.4, vasopressor: 0, fluids: 1, antibiotics: 0, sofa: 2, hours: 3 },
];

//INFERENCE ENGINE (rule-based, clinically grounded mock)
function runInference(p) {
  const sofaNorm = p.sofa / 20;
  const lacNorm = Math.min(p.lactate / 10, 1);
  const ageNorm = (p.age - 18) / 80;
  const mapPenalty = p.map < 65 ? (65 - p.map) / 65 : 0;
  const hrPenalty = p.hr > 100 ? (p.hr - 100) / 60 : 0;

  const baseMortality = Math.min(95, Math.round(
    (sofaNorm * 40 + lacNorm * 25 + ageNorm * 15 + mapPenalty * 12 + hrPenalty * 8) * (1 - p.antibiotics * 0.08 - p.vasopressor * 0.05)
  ));
  const baseShock = Math.min(90, Math.round(
    (mapPenalty * 50 + lacNorm * 30 + sofaNorm * 20) * (1 - p.vasopressor * 0.15)
  ));
  const baseOrgan = Math.min(88, Math.round(
    (p.creatinine / 5 * 30 + sofaNorm * 40 + lacNorm * 15) * (1 - p.fluids * 0.05)
  ));

  const abxEffect = { mortality: -(8 + (p.sofa > 5 ? 5 : 2)), shock: -(3 + (p.lactate > 3 ? 2 : 0)), organ: -(6 + (p.creatinine > 2 ? 3 : 0)) };
  const fluidEffect = { mortality: -(4 + (p.map < 65 ? 4 : 1)), shock: -(10 + (mapPenalty * 8)), organ: -(5 + (p.creatinine > 1.5 ? 2 : 0)) };
  const vasoEffect = { mortality: -(6 + (baseShock > 40 ? 5 : 1)), shock: -(15 + (mapPenalty * 12)), organ: -(4 + (p.map < 65 ? 3 : 0)) };

  const scenarios = [
    { id: "current", label: "Current Protocol", color: C.muted, mortality: baseMortality, shock: baseShock, organ: baseOrgan },
    { id: "abx", label: "Antibiotic Escalation", color: C.green, mortality: Math.max(2, baseMortality + abxEffect.mortality), shock: Math.max(2, baseShock + abxEffect.shock), organ: Math.max(2, baseOrgan + abxEffect.organ), effect: abxEffect },
    { id: "fluid", label: "Fluid Bolus +500mL", color: C.accent, mortality: Math.max(2, baseMortality + fluidEffect.mortality), shock: Math.max(2, baseShock + fluidEffect.shock), organ: Math.max(2, baseOrgan + fluidEffect.organ), effect: fluidEffect },
    { id: "vaso", label: "Vasopressor ↑ Dose", color: C.purple, mortality: Math.max(2, baseMortality + vasoEffect.mortality), shock: Math.max(2, baseShock + vasoEffect.shock), organ: Math.max(2, baseOrgan + vasoEffect.organ), effect: vasoEffect },
  ];

  scenarios.sort((a, b) => a.mortality - b.mortality);
  const best = scenarios[0];

  return { base: { mortality: baseMortality, shock: baseShock, organ: baseOrgan }, scenarios, best };
}

// ─── NAVIGATION ──────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: "home", icon: "⬡", label: "Home" },
  { id: "overview", icon: "◈", label: "Overview" },
  { id: "data", icon: "⊞", label: "Data" },
  { id: "architecture", icon: "⬡", label: "Model" },
  { id: "simulator", icon: "⚕", label: "Simulator" },
  { id: "counterfactual", icon: "⇄", label: "Counterfactual" },
  { id: "performance", icon: "◉", label: "Performance" },
  { id: "training", icon: "⌁", label: "Training" },
  { id: "research", icon: "⊙", label: "Research" },
];

// Web Pages 

// 1. HOME
function HomePage({ onNavigate }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 80);
    return () => clearInterval(id);
  }, []);

  const vitals = [
    { label: "HR", value: 88 + Math.sin(tick * 0.2) * 8, unit: "bpm", color: C.green },
    { label: "MAP", value: 72 + Math.sin(tick * 0.15) * 5, unit: "mmHg", color: C.accent },
    { label: "SpO₂", value: 97 + Math.sin(tick * 0.1) * 1, unit: "%", color: C.teal },
    { label: "Lactate", value: (2.1 + Math.sin(tick * 0.08) * 0.2).toFixed(1), unit: "mmol/L", color: C.yellow },
  ];

  return (
    <div className="fadeIn" style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* HERO */}
      <div style={{ position: "relative", padding: "80px 0 60px", textAlign: "center", overflow: "hidden" }}>
        <div style={{ position: "absolute", inset: 0, background: `radial-gradient(ellipse at 50% 0%, rgba(37,99,235,.18) 0%, transparent 70%)`, pointerEvents: "none" }} />
        {/* ECG line decoration */}
        <svg style={{ position: "absolute", bottom: 0, left: 0, width: "100%", opacity: .12 }} height="60" viewBox="0 0 1200 60">
          <polyline fill="none" stroke={C.accent} strokeWidth="2"
            points="0,30 150,30 170,30 185,5 200,55 215,5 230,55 245,30 400,30 420,30 435,8 450,52 465,8 480,52 495,30 650,30 670,30 685,10 700,50 715,10 730,50 745,30 900,30 920,30 935,8 950,52 965,8 980,52 995,30 1200,30" />
        </svg>
        <div style={{ position: "relative", zIndex: 1 }}>
          <Tag color={C.accent}>ICU Decision Support · v2.1</Tag>
          <h1 style={{ fontSize: "clamp(28px, 5vw, 52px)", fontWeight: 700, fontFamily: fontStack, marginTop: 20, lineHeight: 1.15, color: C.text }}>
            Individualized Sepsis<br /><span style={{ color: C.accent }} className="glow-text">Multi-Treatment Effect</span><br />Prediction
          </h1>
          <p style={{ color: C.muted, fontSize: 16, marginTop: 16, maxWidth: 560, margin: "16px auto 0", lineHeight: 1.7 }}>
            AI-powered causal transformer for personalized ICU treatment planning.<br />
            Counterfactual reasoning meets clinical decision support.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 32, flexWrap: "wrap" }}>
            <button onClick={() => onNavigate("simulator")} style={{ background: C.accentGlow, color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: sansStack }}>
              ⚕ Launch Patient Simulator
            </button>
            <button onClick={() => onNavigate("architecture")} style={{ background: "transparent", color: C.accent, border: `1px solid ${C.accent}`, borderRadius: 8, padding: "12px 28px", fontSize: 14, fontWeight: 600, cursor: "pointer", fontFamily: sansStack }}>
              ⬡ View Architecture
            </button>
            <a href="https://github.com/Emi1ia/EP005-CP--Individualized-Sepsis-Multi-Treatment-Effect-Prediction" target="_blank" rel="noreferrer" style={{ background: "transparent", color: C.muted, border: `1px solid ${C.border}`, borderRadius: 8, padding: "12px 28px", fontSize: 14, fontWeight: 600, cursor: "pointer", textDecoration: "none", fontFamily: sansStack }}>
              ⊙ GitHub Repo
            </a>
          </div>
        </div>
      </div>

      {/* LIVE VITALS STRIP */}
      <div style={{ background: C.surface, borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`, padding: "12px 24px", display: "flex", gap: 32, alignItems: "center", overflowX: "auto" }}>
        <span style={{ fontSize: 10, fontFamily: fontStack, color: C.muted, whiteSpace: "nowrap", letterSpacing: 2 }}>LIVE ICU FEED</span>
        <span className="pulse" style={{ width: 8, height: 8, borderRadius: "50%", background: C.green, display: "inline-block" }} />
        {vitals.map(v => (
          <div key={v.label} style={{ display: "flex", alignItems: "baseline", gap: 6, whiteSpace: "nowrap" }}>
            <span style={{ fontSize: 11, color: C.muted, fontFamily: fontStack }}>{v.label}</span>
            <span style={{ fontSize: 18, fontWeight: 700, color: v.color, fontFamily: fontStack }}>{typeof v.value === "number" ? Math.round(v.value) : v.value}</span>
            <span style={{ fontSize: 10, color: C.muted, fontFamily: fontStack }}>{v.unit}</span>
          </div>
        ))}
      </div>

      {/* KEY STATS */}
      <div style={{ padding: "40px 0", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
        <MetricCard label="Total Patients" value="500" icon="👥" color={C.accent} />
        <MetricCard label="ICU Stays" value="500" icon="🏥" color={C.teal} />
        <MetricCard label="Sepsis Cases" value="500" icon="⚕" color={C.red} />
        <MetricCard label="Clinical Features" value="47" icon="📊" color={C.purple} />
        <MetricCard label="Time-Series Vars" value="18" icon="⌁" color={C.yellow} />
        <MetricCard label="Treatment Arms" value="4" icon="💊" color={C.green} />
      </div>

      {/* HIGHLIGHTS */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginTop: 8 }}>
        {[
          { icon: "⬡", title: "Causal Transformer", desc: "Multi-head attention over clinical time series with propensity weighting for unbiased treatment effect estimation." },
          { icon: "⇄", title: "Counterfactual Reasoning", desc: "Estimates potential outcomes under 4 treatment strategies simultaneously for individualized recommendations." },
          { icon: "◈", title: "Multi-Task Learning", desc: "Joint optimization for mortality prediction, shock progression, and organ dysfunction with shared representations." },
          { icon: "⊙", title: "MIMIC-III / eICU Ready", desc: "Preprocessing pipelines for both MIMIC-III and eICU databases with automated feature engineering." },
        ].map(h => (
          <div key={h.title} style={{ ...card(), padding: 24 }}>
            <div style={{ fontSize: 28, marginBottom: 12 }}>{h.icon}</div>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: C.text, marginBottom: 8, fontFamily: fontStack }}>{h.title}</h3>
            <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.7 }}>{h.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// 2. OVERVIEW
function OverviewPage() {
  const stats = [
    { label: "Total Patients", value: "", desc: "Adult ICU admissions", color: C.accent },
    { label: "Sepsis Cases", value: "500", desc: "Sepsis-3 criteria", color: C.red },
    { label: "Avg SOFA Score", value: "5.2", desc: "Severity of illness", color: C.yellow },
    { label: "28-Day Mortality", value: "23.4%", desc: "Primary outcome", color: C.purple },
    { label: "Features Used", value: "47", desc: "Clinical variables", color: C.teal },
    { label: "Time-Series Vars", value: "18", desc: "Hourly measurements", color: C.green },
  ];
  return (
    <div className="fadeIn">
      <SectionHeader title="Dataset Overview" subtitle="Summary statistics from the synthetic ICU dataset based on MIMIC-III schema" badge="500 patients" />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14, marginBottom: 32 }}>
        {stats.map(s => (
          <div key={s.label} style={{ ...card(), padding: 18 }}>
            <div style={{ fontSize: 11, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{s.label}</div>
            <div style={{ fontSize: 30, fontWeight: 700, color: s.color, fontFamily: fontStack }}>{s.value}</div>
            <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{s.desc}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={card()}>
          <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Age Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={ageDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="age" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              <Bar dataKey="count" fill={C.accent} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={card()}>
          <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Mortality by Age Group</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={mortalityByAge}>
              <defs><linearGradient id="mg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.red} stopOpacity={0.3} /><stop offset="95%" stopColor={C.red} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="age" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} formatter={v => [v + "%", "Mortality"]} />
              <Area type="monotone" dataKey="mortality" stroke={C.red} fill="url(#mg)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div style={card()}>
        <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Treatment Frequency</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {treatmentFreq.map(t => (
            <div key={t.name} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 130, fontSize: 12, color: C.text }}>{t.name}</div>
              <div style={{ flex: 1, background: C.border, borderRadius: 4, height: 8, overflow: "hidden" }}>
                <div style={{ width: `${t.pct}%`, height: "100%", background: `linear-gradient(90deg, ${C.accent}, ${C.accentGlow})`, borderRadius: 4, transition: "width 1s" }} />
              </div>
              <div style={{ width: 50, textAlign: "right", fontSize: 12, fontFamily: fontStack, color: C.accent }}>{t.pct}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// 3. DATA EXPLORATION
function DataPage() {
  const organData = [
    { name: "Renal", d1: 28, d2: 35, d3: 42 }, { name: "Hepatic", d1: 18, d2: 24, d3: 30 },
    { name: "Coagulation", d1: 22, d2: 29, d3: 35 }, { name: "Cardiovascular", d1: 35, d2: 48, d3: 57 },
    { name: "Neurologic", d1: 12, d2: 18, d3: 24 }, { name: "Respiratory", d1: 40, d2: 52, d3: 61 },
  ];
  const genderData = [{ name: "Male", value: 54 }, { name: "Female", value: 44 }, { name: "Other", value: 2 }];
  const gColors = [C.accent, C.purple, C.teal];

  return (
    <div className="fadeIn">
      <SectionHeader title="Data Exploration" subtitle="Distribution analysis and clinical feature statistics from ICU cohort" badge="Synthetic Data" />
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={card()}>
          <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Gender Distribution</h3>
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <ResponsiveContainer width={160} height={160}>
              <PieChart>
                <Pie data={genderData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} dataKey="value" paddingAngle={3}>
                  {genderData.map((_, i) => <Cell key={i} fill={gColors[i]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ flex: 1 }}>
              {genderData.map((g, i) => (
                <div key={g.name} style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, alignItems: "center" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: gColors[i] }} />
                    <span style={{ fontSize: 13, color: C.text }}>{g.name}</span>
                  </div>
                  <span style={{ fontSize: 14, fontWeight: 700, color: gColors[i], fontFamily: fontStack }}>{g.value}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div style={card()}>
          <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Shock Progression (by hour)</h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={shockProgression}>
              <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.yellow} stopOpacity={0.4} /><stop offset="95%" stopColor={C.yellow} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="hour" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} label={{ value: "hrs", position: "insideRight", fill: C.muted, fontSize: 10 }} />
              <YAxis tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} formatter={v => [v + "%", "Shock Risk"]} />
              <Area type="monotone" dataKey="risk" stroke={C.yellow} fill="url(#sg)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div style={card()}>
        <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Organ Dysfunction Trends (Day 1 / Day 2 / Day 3)</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={organData}>
            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
            <XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
            <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
            <Bar dataKey="d1" name="Day 1" fill={C.accent} radius={[3, 3, 0, 0]} opacity={0.8} />
            <Bar dataKey="d2" name="Day 2" fill={C.yellow} radius={[3, 3, 0, 0]} opacity={0.8} />
            <Bar dataKey="d3" name="Day 3" fill={C.red} radius={[3, 3, 0, 0]} opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ ...card(), marginTop: 16 }}>
        <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 14, textTransform: "uppercase", letterSpacing: 1 }}>Missing Value Profile</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8 }}>
          {[
            { feat: "Heart Rate", miss: 2 }, { feat: "MAP", miss: 4 }, { feat: "Temperature", miss: 8 },
            { feat: "Creatinine", miss: 11 }, { feat: "Lactate", miss: 16 }, { feat: "WBC", miss: 7 },
            { feat: "Platelet", miss: 12 }, { feat: "Bilirubin", miss: 19 }, { feat: "GCS", miss: 6 },
            { feat: "SpO₂", miss: 3 }, { feat: "Resp Rate", miss: 5 }, { feat: "Glucose", miss: 9 },
          ].map(f => (
            <div key={f.feat} style={{ background: C.surface, borderRadius: 6, padding: 10, border: `1px solid ${C.border}` }}>
              <div style={{ fontSize: 11, color: C.text, marginBottom: 6, fontFamily: fontStack }}>{f.feat}</div>
              <div style={{ height: 4, background: C.border, borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${f.miss}%`, height: "100%", background: f.miss > 15 ? C.red : f.miss > 10 ? C.yellow : C.green }} />
              </div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontFamily: fontStack }}>{f.miss}% missing</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// 4. MODEL ARCHITECTURE
function ArchitecturePage() {
  const [expanded, setExpanded] = useState(null);
  const blocks = [
    { id: "input", label: "Input Layer", color: C.teal, icon: "⊞", detail: "Concatenation of 47 clinical features including 18 time-series variables at hourly resolution. Includes demographic, lab, vital sign, and treatment history inputs. Sequence length: T=24 hours." },
    { id: "embed", label: "Temporal Embedding", color: C.accent, icon: "⌁", detail: "Positional + value encoding. Each time step is projected to d_model=128 dimensions. Learnable positional embeddings capture temporal ordering. Treatment history encoded as categorical + continuous features." },
    { id: "encoder", label: "Transformer Encoder", color: C.purple, icon: "⬡", detail: "4 stacked transformer encoder blocks. Each block: Multi-Head Self-Attention (8 heads, d_k=16) → LayerNorm → Feed-Forward (512→128) → Dropout(0.1) → LayerNorm. Total: ~850K parameters." },
    { id: "propensity", label: "Propensity Head", color: C.yellow, icon: "◉", detail: "Estimates treatment propensity scores P(T|X) for inverse probability weighting. Separate 2-layer MLP for each of 4 treatments. Output: softmax probabilities. Used to debias observational confounding." },
    { id: "decoder", label: "Counterfactual Decoder", color: C.red, icon: "⇄", detail: "For each treatment arm, a dedicated decoder projects the shared representation through a treatment-specific MLP (128→64→32). Outputs potential outcomes Y(a) for a ∈ {0,1,2,3} simultaneously." },
    { id: "output", label: "Output / Recommendation", color: C.green, icon: "⚕", detail: "Multi-task outputs: (1) 28-day mortality risk, (2) 24h shock progression, (3) organ dysfunction score. Best treatment selected by minimum ITE-weighted composite score. CATE estimated as E[Y(a) - Y(0)|X]." },
  ];
  const params = [
    { name: "d_model", val: "128" }, { name: "n_heads", val: "8" }, { name: "n_layers", val: "4" },
    { name: "ff_dim", val: "512" }, { name: "dropout", val: "0.1" }, { name: "seq_len", val: "24" },
    { name: "lr", val: "1e-4" }, { name: "batch_size", val: "64" }, { name: "epochs", val: "50" },
    { name: "optimizer", val: "AdamW" }, { name: "loss", val: "BCE + MSE" }, { name: "params", val: "~1.2M" },
  ];

  return (
    <div className="fadeIn">
      <SectionHeader title="Model Architecture" subtitle="Causal Transformer for multi-treatment effect estimation in ICU settings" badge="~1.2M Parameters" />
      {/* Architecture flow */}
      <div style={{ ...card(), padding: 24, marginBottom: 16 }}>
        <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 20, textTransform: "uppercase", letterSpacing: 1 }}>Pipeline Flow — Click for details</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {blocks.map((b, i) => (
            <div key={b.id}>
              <div onClick={() => setExpanded(expanded === b.id ? null : b.id)}
                style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 18px", background: expanded === b.id ? b.color + "18" : C.surface, border: `1px solid ${expanded === b.id ? b.color : C.border}`, borderRadius: 8, cursor: "pointer", transition: "all .2s" }}>
                <div style={{ width: 36, height: 36, borderRadius: 8, background: b.color + "20", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, color: b.color, border: `1px solid ${b.color}40` }}>{b.icon}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: C.text, fontFamily: fontStack }}>{b.label}</div>
                </div>
                <div style={{ color: b.color, fontSize: 12, fontFamily: fontStack }}>{expanded === b.id ? "▲ collapse" : "▼ expand"}</div>
              </div>
              {expanded === b.id && (
                <div style={{ background: b.color + "08", border: `1px solid ${b.color}30`, borderTop: "none", borderRadius: "0 0 8px 8px", padding: "14px 18px 14px 68px" }}>
                  <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.8 }}>{b.detail}</p>
                </div>
              )}
              {i < blocks.length - 1 && (
                <div style={{ display: "flex", justifyContent: "center", padding: "4px 0" }}>
                  <div style={{ width: 1, height: 20, background: `linear-gradient(${b.color}, ${blocks[i + 1].color})` }} />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      {/* Hyperparams */}
      <div style={card()}>
        <h3 style={{ fontSize: 13, color: C.muted, fontFamily: fontStack, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Training Hyperparameters</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10 }}>
          {params.map(p => (
            <div key={p.name} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 14px" }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1 }}>{p.name}</div>
              <div style={{ fontSize: 16, fontWeight: 600, color: C.accent, fontFamily: fontStack, marginTop: 4 }}>{p.val}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// 5. PATIENT SIMULATOR
function SimulatorPage() {
  const defaultPatient = { age: 62, hr: 104, map: 62, lactate: 3.2, creatinine: 1.8, wbc: 14.6, temp: 38.5, vasopressor: 0, fluids: 1, antibiotics: 1, sofa: 6, hours: 8 };
  const [patient, setPatient] = useState(defaultPatient);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadPreset = (p) => { const { name, ...rest } = p; setPatient(rest); setResult(null); };

  const runSimulation = () => {
    setLoading(true);
    setTimeout(() => {
      setResult(runInference(patient));
      setLoading(false);
    }, 900);
  };

  const Field = ({ label, field, min, max, step = 1, unit = "" }) => (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <label style={{ fontSize: 11, color: C.muted, fontFamily: fontStack }}>{label}</label>
        <span style={{ fontSize: 12, color: C.accent, fontFamily: fontStack }}>{patient[field]}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={patient[field]}
        onChange={e => setPatient(p => ({ ...p, [field]: +e.target.value }))} style={{ width: "100%" }} />
    </div>
  );

  const Toggle = ({ label, field }) => (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0" }}>
      <span style={{ fontSize: 12, color: C.muted, fontFamily: fontStack }}>{label}</span>
      <button onClick={() => setPatient(p => ({ ...p, [field]: p[field] === 1 ? 0 : 1 }))}
        style={{ background: patient[field] ? C.green + "20" : C.border, color: patient[field] ? C.green : C.muted, border: `1px solid ${patient[field] ? C.green : C.border}`, borderRadius: 20, padding: "4px 14px", fontSize: 11, cursor: "pointer", fontFamily: fontStack }}>
        {patient[field] ? "YES" : "NO"}
      </button>
    </div>
  );

  return (
    <div className="fadeIn">
      <SectionHeader title="Patient Simulator" subtitle="Enter or adjust patient parameters and run causal inference to get individualized treatment recommendations" badge="Inference Engine" />
      {/* Presets */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
        {DEMO_PATIENTS.map(dp => (
          <button key={dp.name} onClick={() => loadPreset(dp)}
            style={{ background: C.surface, color: C.muted, border: `1px solid ${C.border}`, borderRadius: 6, padding: "6px 14px", fontSize: 11, cursor: "pointer", fontFamily: fontStack }}>
            {dp.name}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 16 }}>
        {/* Input Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ ...card(), padding: 18 }}>
            <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Vitals & Labs</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <Field label="Age" field="age" min={18} max={90} unit=" yrs" />
              <Field label="Heart Rate" field="hr" min={40} max={160} unit=" bpm" />
              <Field label="MAP" field="map" min={40} max={120} unit=" mmHg" />
              <Field label="Lactate" field="lactate" min={0.5} max={12} step={0.1} unit=" mmol/L" />
              <Field label="Creatinine" field="creatinine" min={0.4} max={8} step={0.1} unit=" mg/dL" />
              <Field label="WBC" field="wbc" min={2} max={40} step={0.1} unit=" ×10³/μL" />
              <Field label="Temperature" field="temp" min={35} max={41} step={0.1} unit=" °C" />
            </div>
          </div>
          <div style={{ ...card(), padding: 18 }}>
            <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Severity & Treatment</h3>
            <Field label="SOFA Score" field="sofa" min={0} max={20} unit="" />
            <Field label="Hours since sepsis onset" field="hours" min={0} max={72} unit=" hrs" />
            <div style={{ borderTop: `1px solid ${C.border}`, margin: "10px 0" }} />
            <Toggle label="Antibiotics active" field="antibiotics" />
            <Toggle label="Vasopressor support" field="vasopressor" />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 0" }}>
              <span style={{ fontSize: 12, color: C.muted, fontFamily: fontStack }}>Fluid bolus level</span>
              <div style={{ display: "flex", gap: 4 }}>
                {[0, 1, 2, 3].map(l => (
                  <button key={l} onClick={() => setPatient(p => ({ ...p, fluids: l }))}
                    style={{ width: 28, height: 28, borderRadius: 4, background: patient.fluids === l ? C.accent + "30" : C.surface, color: patient.fluids === l ? C.accent : C.muted, border: `1px solid ${patient.fluids === l ? C.accent : C.border}`, fontSize: 11, cursor: "pointer" }}>
                    {l}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <button onClick={runSimulation} disabled={loading}
            style={{ background: loading ? C.border : `linear-gradient(135deg, ${C.accentGlow}, ${C.accent})`, color: "#fff", border: "none", borderRadius: 8, padding: "14px", fontSize: 14, fontWeight: 700, cursor: loading ? "wait" : "pointer", fontFamily: fontStack, letterSpacing: 1 }}>
            {loading ? "⌁ RUNNING INFERENCE..." : "⚕ RUN INFERENCE"}
          </button>
        </div>

        {/* Results Panel */}
        <div>
          {!result && !loading && (
            <div style={{ ...card(), height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12, minHeight: 400 }}>
              <div style={{ fontSize: 48, opacity: .3 }}>⚕</div>
              <div style={{ color: C.muted, fontSize: 14, fontFamily: fontStack }}>Configure patient and run inference</div>
            </div>
          )}
          {loading && (
            <div style={{ ...card(), height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 16, minHeight: 400 }}>
              <div className="pulse" style={{ fontSize: 48 }}>⬡</div>
              <div style={{ color: C.accent, fontSize: 13, fontFamily: fontStack }}>Computing counterfactuals...</div>
              <div style={{ color: C.muted, fontSize: 11 }}>Propensity estimation · Treatment effects · Recommendations</div>
            </div>
          )}
          {result && !loading && (
            <div className="fadeIn" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Risk Gauges */}
              <div style={{ ...card("glow"), padding: 20 }}>
                <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Predicted Risk (Next 24h — Current Protocol)</h3>
                <div style={{ display: "flex", justifyContent: "space-around" }}>
                  <RiskGauge value={result.base.mortality} label="28-Day Mortality" color={result.base.mortality > 50 ? C.red : result.base.mortality > 30 ? C.yellow : C.green} />
                  <RiskGauge value={result.base.shock} label="Shock Progression" color={result.base.shock > 50 ? C.red : result.base.shock > 30 ? C.yellow : C.green} />
                  <RiskGauge value={result.base.organ} label="Organ Dysfunction" color={result.base.organ > 50 ? C.red : result.base.organ > 30 ? C.yellow : C.green} />
                </div>
              </div>
              {/* Best Recommendation */}
              <div style={{ background: result.best.color + "12", border: `1px solid ${result.best.color}`, borderRadius: 12, padding: 20 }}>
                <div style={{ fontSize: 10, color: result.best.color, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 2, marginBottom: 8 }}>★ Recommended Treatment Strategy</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: result.best.color, fontFamily: fontStack }}>{result.best.label}</div>
                {result.best.effect && (
                  <div style={{ display: "flex", gap: 16, marginTop: 10 }}>
                    {[
                      { label: "Mortality Δ", val: result.best.effect.mortality },
                      { label: "Shock Δ", val: result.best.effect.shock },
                      { label: "Organ Δ", val: result.best.effect.organ },
                    ].map(e => (
                      <div key={e.label} style={{ background: result.best.color + "15", borderRadius: 6, padding: "8px 12px", textAlign: "center" }}>
                        <div style={{ fontSize: 10, color: C.muted, fontFamily: fontStack }}>{e.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 700, color: e.val < 0 ? C.green : C.red, fontFamily: fontStack }}>{e.val > 0 ? "+" : ""}{e.val}%</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {/* Treatment comparison chart */}
              <div style={card()}>
                <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Treatment Effect Comparison — Mortality Risk</h3>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={result.scenarios} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
                    <XAxis type="number" tick={{ fill: C.muted, fontSize: 11 }} unit="%" domain={[0, 100]} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="label" tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} width={160} />
                    <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} formatter={v => [v + "%", "Mortality Risk"]} />
                    <Bar dataKey="mortality" radius={[0, 4, 4, 0]}>
                      {result.scenarios.map((s) => <Cell key={s.id} fill={s.color} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// 6. COUNTERFACTUAL ANALYSIS
function CounterfactualPage() {
  const [patientPreset, setPatientPreset] = useState(0);
  const p = DEMO_PATIENTS[patientPreset];
  const { name, ...pat } = p;
  const result = runInference(pat);

  const radarData = result.scenarios.map(s => ({
    subject: s.label.split(" ").slice(0, 2).join(" "),
    mortality: s.mortality, shock: s.shock, organ: s.organ,
  }));

  return (
    <div className="fadeIn">
      <SectionHeader title="Counterfactual Analysis" subtitle="Compare predicted outcomes across all treatment scenarios for the same patient" badge="ITE Estimation" />
      <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
        {DEMO_PATIENTS.map((dp, i) => (
          <button key={dp.name} onClick={() => setPatientPreset(i)}
            style={{ background: patientPreset === i ? C.accent + "20" : C.surface, color: patientPreset === i ? C.accent : C.muted, border: `1px solid ${patientPreset === i ? C.accent : C.border}`, borderRadius: 6, padding: "6px 14px", fontSize: 11, cursor: "pointer", fontFamily: fontStack }}>
            {dp.name}
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Outcome Comparison — All Scenarios</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={result.scenarios}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="label" tick={false} axisLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }}
                labelFormatter={(_, payload) => payload?.[0]?.payload?.label || ""}
                formatter={(v, n) => [v + "%", n === "mortality" ? "Mortality" : n === "shock" ? "Shock" : "Organ Dysfunction"]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="mortality" name="mortality" radius={[3, 3, 0, 0]}>{result.scenarios.map(s => <Cell key={s.id} fill={s.color} />)}</Bar>
              <Bar dataKey="shock" name="shock" fill={C.yellow} opacity={0.6} radius={[3, 3, 0, 0]} />
              <Bar dataKey="organ" name="organ" fill={C.purple} opacity={0.5} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Radar — Outcome Profile per Treatment</h3>
          <ResponsiveContainer width="100%" height={230}>
            <RadarChart data={[
              { metric: "Mortality", ...Object.fromEntries(result.scenarios.map(s => [s.id, s.mortality])) },
              { metric: "Shock", ...Object.fromEntries(result.scenarios.map(s => [s.id, s.shock])) },
              { metric: "Organ Fail", ...Object.fromEntries(result.scenarios.map(s => [s.id, s.organ])) },
            ]}>
              <PolarGrid stroke={C.border} />
              <PolarAngleAxis dataKey="metric" tick={{ fill: C.muted, fontSize: 11 }} />
              {result.scenarios.map(s => (
                <Radar key={s.id} name={s.label} dataKey={s.id} stroke={s.color} fill={s.color} fillOpacity={0.1} />
              ))}
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ITE Table */}
      <div style={card()}>
        <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Individual Treatment Effects (ITE) vs. Current Protocol</h3>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                {["Scenario", "Mortality Risk", "Δ vs Current", "Shock Risk", "Δ vs Current", "Organ Risk", "Δ vs Current", "Rank"].map(h => (
                  <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontSize: 10, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.scenarios.map((s, i) => {
                const base = result.scenarios.find(x => x.id === "current") || result.scenarios[0];
                const dM = s.mortality - base.mortality;
                const dS = s.shock - base.shock;
                const dO = s.organ - base.organ;
                return (
                  <tr key={s.id} style={{ borderBottom: `1px solid ${C.border}20`, background: i === 0 && s.id !== "current" ? s.color + "08" : "transparent" }}>
                    <td style={{ padding: "10px 12px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.color }} />
                        <span style={{ color: C.text }}>{s.label}</span>
                      </div>
                    </td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: s.color }}>{s.mortality}%</td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: dM < 0 ? C.green : dM > 0 ? C.red : C.muted }}>{dM === 0 ? "—" : (dM > 0 ? "+" : "") + dM + "%"}</td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: C.muted }}>{s.shock}%</td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: dS < 0 ? C.green : dS > 0 ? C.red : C.muted }}>{dS === 0 ? "—" : (dS > 0 ? "+" : "") + dS + "%"}</td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: C.muted }}>{s.organ}%</td>
                    <td style={{ padding: "10px 12px", fontFamily: fontStack, color: dO < 0 ? C.green : dO > 0 ? C.red : C.muted }}>{dO === 0 ? "—" : (dO > 0 ? "+" : "") + dO + "%"}</td>
                    <td style={{ padding: "10px 12px" }}>
                      {i === 0 ? <Tag color={C.green}>★ Best</Tag> : <span style={{ color: C.muted, fontSize: 12 }}>#{i + 1}</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// 7. MODEL PERFORMANCE
function PerformancePage() {
  const metricsClass = [
    { name: "AUROC — Mortality", value: 0.847, color: C.accent },
    { name: "AUROC — Shock", value: 0.813, color: C.yellow },
    { name: "AUPRC — Mortality", value: 0.621, color: C.teal },
    { name: "F1-Score — Mortality", value: 0.668, color: C.purple },
  ];
  const metricsCausal = [
    { name: "PEHE (√)", value: "4.21", unit: "%", color: C.green, note: "Lower is better" },
    { name: "ATE Error", value: "2.13", unit: "%", color: C.accent, note: "Lower is better" },
    { name: "Policy Value", value: "0.831", unit: "", color: C.teal, note: "Higher is better" },
  ];
  const featureImportance = [
    { feat: "SOFA Score", imp: 92 }, { feat: "Lactate", imp: 87 }, { feat: "MAP", imp: 81 },
    { feat: "Creatinine", imp: 74 }, { feat: "HR (trend)", imp: 68 }, { feat: "Age", imp: 62 },
    { feat: "WBC", imp: 55 }, { feat: "Temperature", imp: 48 }, { feat: "Antibiotics history", imp: 43 },
    { feat: "Fluid balance", imp: 38 },
  ];

  return (
    <div className="fadeIn">
      <SectionHeader title="Model Performance" subtitle="Classification, regression, and causal evaluation metrics" badge="Held-out Test Set" />
      <div style={{ background: C.yellow + "10", border: `1px solid ${C.yellow}40`, borderRadius: 8, padding: "10px 16px", marginBottom: 20, fontSize: 12, color: C.yellow }}>
        ⚠ Note: Metrics shown are from synthetic validation data. Run training pipeline for final results.
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 20 }}>
        {metricsClass.map(m => (
          <div key={m.name} style={{ ...card(), padding: 18 }}>
            <div style={{ fontSize: 11, color: C.muted, fontFamily: fontStack, marginBottom: 8 }}>{m.name}</div>
            <div style={{ fontSize: 34, fontWeight: 700, color: m.color, fontFamily: fontStack }}>{m.value}</div>
            <div style={{ height: 4, background: C.border, borderRadius: 2, overflow: "hidden", marginTop: 10 }}>
              <div style={{ width: `${m.value * 100}%`, height: "100%", background: m.color, borderRadius: 2 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>ROC Curve — Mortality</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="fpr" tick={{ fill: C.muted, fontSize: 10 }} label={{ value: "FPR", position: "insideBottom", offset: -4, fill: C.muted, fontSize: 10 }} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }} label={{ value: "TPR", angle: -90, position: "insideLeft", fill: C.muted, fontSize: 10 }} />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 }} />
              <Line type="monotone" dataKey="tpr" stroke={C.accent} strokeWidth={2} dot={false} name="Model (AUC=0.847)" />
              <Line type="monotone" data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke={C.border} strokeWidth={1} dot={false} strokeDasharray="4 4" name="Random" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Feature Importance (SHAP-like)</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {featureImportance.map(f => (
              <div key={f.feat} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 110, fontSize: 11, color: C.text, flexShrink: 0 }}>{f.feat}</div>
                <div style={{ flex: 1, background: C.border, borderRadius: 3, height: 6, overflow: "hidden" }}>
                  <div style={{ width: `${f.imp}%`, height: "100%", background: `linear-gradient(90deg, ${C.purple}, ${C.accent})`, borderRadius: 3 }} />
                </div>
                <div style={{ width: 32, textAlign: "right", fontSize: 11, fontFamily: fontStack, color: C.accent }}>{f.imp}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        {metricsCausal.map(m => (
          <div key={m.name} style={{ ...card(), padding: 18 }}>
            <div style={{ fontSize: 11, color: C.muted, fontFamily: fontStack, marginBottom: 6 }}>{m.name}</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: m.color, fontFamily: fontStack }}>{m.value}<span style={{ fontSize: 14 }}>{m.unit}</span></div>
            <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontFamily: fontStack }}>{m.note}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// 8. TRAINING MONITOR
function TrainingPage() {
  const [epoch, setEpoch] = useState(50);
  const displayed = lossData.slice(0, epoch);
  const auroc = Array.from({ length: 50 }, (_, i) => ({
    epoch: i + 1,
    auroc: +(0.5 + (0.847 - 0.5) * (1 - Math.exp(-0.1 * (i + 1))) + (Math.random() - 0.5) * 0.02).toFixed(3),
  }));
  const lrData = Array.from({ length: 50 }, (_, i) => ({
    epoch: i + 1,
    lr: +(1e-4 * (i < 20 ? 1 : Math.exp(-0.05 * (i - 20)))).toExponential(2),
  }));

  return (
    <div className="fadeIn">
      <SectionHeader title="Training Monitor" subtitle="Loss curves, validation metrics, and learning rate schedule" badge="50 Epochs" />
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 20 }}>
        <span style={{ fontSize: 12, color: C.muted, fontFamily: fontStack }}>Replay to epoch:</span>
        <input type="range" min={1} max={50} value={epoch} onChange={e => setEpoch(+e.target.value)} style={{ width: 200 }} />
        <span style={{ fontSize: 14, color: C.accent, fontFamily: fontStack, width: 30 }}>{epoch}</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Training & Validation Loss</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={displayed}>
              <defs>
                <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.accent} stopOpacity={0.15} /><stop offset="95%" stopColor={C.accent} stopOpacity={0} /></linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="epoch" tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train" stroke={C.accent} strokeWidth={2} dot={false} name="Train Loss" />
              <Line type="monotone" dataKey="val" stroke={C.red} strokeWidth={2} dot={false} name="Val Loss" strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={card()}>
          <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>Validation AUROC</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={auroc.slice(0, epoch)}>
              <defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={C.green} stopOpacity={0.25} /><stop offset="95%" stopColor={C.green} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
              <XAxis dataKey="epoch" tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.5, 1]} tick={{ fill: C.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: C.cardBright, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 }} />
              <Area type="monotone" dataKey="auroc" stroke={C.green} fill="url(#ag)" strokeWidth={2} name="AUROC" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { label: "Best AUROC", val: "0.847", epoch: "Ep 47", color: C.green },
          { label: "Early Stop", val: "Ep 50", epoch: "patience=5", color: C.accent },
          { label: "Final Train Loss", val: "0.184", epoch: "", color: C.yellow },
          { label: "Final Val Loss", val: "0.231", epoch: "", color: C.red },
        ].map(m => (
          <div key={m.label} style={{ ...card(), padding: 16 }}>
            <div style={{ fontSize: 10, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>{m.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: m.color, fontFamily: fontStack }}>{m.val}</div>
            <div style={{ fontSize: 10, color: C.muted, marginTop: 4, fontFamily: fontStack }}>{m.epoch}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// 9. RESEARCH
function ResearchPage() {
  const refs = [
    "Vaswani et al. (2017). Attention Is All You Need. NeurIPS.",
    "Bica et al. (2020). Estimating ITE with Time-Varying Confounders. ICDM.",
    "Rubin, D.B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies.",
    "Singer et al. (2016). Sepsis-3 Definitions. JAMA.",
    "Johnson et al. (2016). MIMIC-III: A Freely Accessible Critical Care Database. Scientific Data.",
    "Komorowski et al. (2018). The AI Clinician for Sepsis. Nature Medicine.",
    "Lim et al. (2018). Forecasting Treatment Responses. NeurIPS.",
  ];

  return (
    <div className="fadeIn" style={{ maxWidth: 760 }}>
      <SectionHeader title="Research Foundation" subtitle="Scientific motivation and contributions of this project" />
      {[
        {
          title: "Problem Statement", color: C.red,
          content: "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. It affects over 49 million people annually and is the leading cause of ICU mortality worldwide. Despite its prevalence, optimal treatment remains elusive — largely because heterogeneous patient populations respond differently to interventions like antibiotics, vasopressors, and fluid resuscitation.",
        },
        {
          title: "Why Causal ML?", color: C.accent,
          content: "Standard predictive models estimate P(outcome | patient). Causal models estimate P(outcome | patient, do(treatment)), enabling counterfactual reasoning: 'What would have happened if we had given vasopressors instead of fluids?' This distinction is critical for treatment recommendations. Observational confounding — where sicker patients receive more aggressive treatment — must be corrected via propensity weighting or doubly robust estimators.",
        },
        {
          title: "Research Gap", color: C.yellow,
          content: "Prior work (Komorowski 2018, AI Clinician) used reinforcement learning but single-treatment framing. Recent causal transformer work (Bica 2020) addressed time-varying confounding but not multi-treatment simultaneous estimation. This project bridges that gap: a unified causal transformer estimating individualized treatment effects for 4 treatment arms simultaneously from ICU time-series data.",
        },
        {
          title: "Technical Contributions", color: C.green,
          content: "1) Causal transformer architecture with treatment-specific decoder heads. 2) Multi-task learning for mortality, shock, and organ dysfunction. 3) Propensity score integration for deconfounding. 4) Counterfactual recommendation engine. 5) Comprehensive evaluation: AUROC, PEHE, ATE error, policy value.",
        },
      ].map(s => (
        <div key={s.title} style={{ ...card(), padding: 22, marginBottom: 14, borderLeft: `3px solid ${s.color}` }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, color: s.color, fontFamily: fontStack, marginBottom: 10 }}>{s.title}</h3>
          <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.9 }}>{s.content}</p>
        </div>
      ))}
      <div style={{ ...card(), padding: 22 }}>
        <h3 style={{ fontSize: 12, color: C.muted, fontFamily: fontStack, textTransform: "uppercase", letterSpacing: 1, marginBottom: 14 }}>References</h3>
        {refs.map((r, i) => (
          <div key={i} style={{ display: "flex", gap: 12, marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: C.accent, fontFamily: fontStack, minWidth: 24 }}>[{i + 1}]</span>
            <span style={{ fontSize: 12, color: C.muted, lineHeight: 1.7 }}>{r}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// MAIN APLICATION
export default function App() {
  const [page, setPage] = useState("home");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const pages = {
    home: <HomePage onNavigate={setPage} />,
    overview: <OverviewPage />,
    data: <DataPage />,
    architecture: <ArchitecturePage />,
    simulator: <SimulatorPage />,
    counterfactual: <CounterfactualPage />,
    performance: <PerformancePage />,
    training: <TrainingPage />,
    research: <ResearchPage />,
  };

  return (
    <>
      <GlobalStyle />
      <div style={{ display: "flex", minHeight: "100vh", fontFamily: sansStack }}>
        {/* SIDEBAR */}
        <div style={{
          width: sidebarOpen ? 220 : 56, background: C.surface, borderRight: `1px solid ${C.border}`,
          display: "flex", flexDirection: "column", transition: "width .25s ease", flexShrink: 0,
          position: "sticky", top: 0, height: "100vh", overflowY: "auto", overflowX: "hidden",
        }}>
          {/* Logo */}
          <div style={{ padding: sidebarOpen ? "20px 16px 16px" : "20px 12px 16px", borderBottom: `1px solid ${C.border}`, display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}
            onClick={() => setSidebarOpen(o => !o)}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: `linear-gradient(135deg, ${C.accentGlow}, ${C.accent})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0 }}>⚕</div>
            {sidebarOpen && <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.text, fontFamily: fontStack, lineHeight: 1 }}>SEPSIS</div>
              <div style={{ fontSize: 9, color: C.muted, fontFamily: fontStack, letterSpacing: 1 }}>DECISION AI</div>
            </div>}
          </div>

          {/* Nav */}
          <nav style={{ flex: 1, padding: "12px 8px" }}>
            {NAV_ITEMS.map(item => (
              <button key={item.id} onClick={() => setPage(item.id)}
                style={{
                  display: "flex", alignItems: "center", gap: 10, width: "100%", padding: sidebarOpen ? "10px 10px" : "10px", marginBottom: 2,
                  background: page === item.id ? C.accent + "18" : "transparent", color: page === item.id ? C.accent : C.muted,
                  border: `1px solid ${page === item.id ? C.accent + "40" : "transparent"}`, borderRadius: 8, cursor: "pointer", textAlign: "left",
                  fontSize: 13, fontFamily: sansStack, fontWeight: page === item.id ? 600 : 400, transition: "all .15s",
                }}>
                <span style={{ fontSize: 16, flexShrink: 0 }}>{item.icon}</span>
                {sidebarOpen && <span style={{ whiteSpace: "nowrap" }}>{item.label}</span>}
              </button>
            ))}
          </nav>

          {/* Footer */}
          {sidebarOpen && (
            <div style={{ padding: "14px 16px", borderTop: `1px solid ${C.border}` }}>
              <div style={{ fontSize: 10, color: C.muted, fontFamily: fontStack, lineHeight: 1.7 }}>
                EP005-CP Project<br />Causal Transformer v2<br />
                <span style={{ color: C.accent }}>● Synthetic demo data</span>
              </div>
            </div>
          )}
        </div>

        {/* MAIN CONTENT */}
        <div style={{ flex: 1, padding: "32px 40px", overflowY: "auto", maxHeight: "100vh" }}>
          <div style={{ maxWidth: 1100, margin: "0 auto" }}>
            {pages[page]}
          </div>
        </div>
      </div>
    </>
  );
}
