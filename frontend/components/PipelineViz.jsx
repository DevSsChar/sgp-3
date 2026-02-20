"use client";
import { useEffect, useState } from "react";
import { TreePine, Zap, BarChart3 } from 'lucide-react';

export default function PipelineViz() {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 60);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="relative w-full h-[520px] select-none">
      {/* Radial glow behind */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div
          className="w-[420px] h-[420px] rounded-full"
          style={{
            background: "radial-gradient(circle, rgba(82,82,82,0.25) 0%, transparent 70%)",
          }}
        />
      </div>

      {/* === CSV Card === */}
      <div
        className="absolute glass-card rounded-2xl p-4 w-[152px] animate-float"
        style={{ top: 20, left: 0 }}
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: "rgba(117,117,117,0.5)" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#B0B0B0" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
          </div>
          <span className="text-xs font-semibold" style={{ color: "#E0E0E0" }}>dataset.csv</span>
        </div>
        <div className="space-y-1">
          {["col_a,col_b", "1.0, 0.3", "2.1, 0.7", "0.8, 1.2"].map((r, i) => (
            <div key={i} className="text-[9px] font-mono px-1 rounded" style={{ color: "#8C8C8C", background: "rgba(82,82,82,0.3)" }}>{r}</div>
          ))}
        </div>
        <div className="mt-2 flex gap-1">
          <span className="text-[8px] px-1.5 py-0.5 rounded" style={{ background: "rgba(117,117,117,0.4)", color: "#B0B0B0" }}>10k rows</span>
          <span className="text-[8px] px-1.5 py-0.5 rounded" style={{ background: "rgba(117,117,117,0.4)", color: "#B0B0B0" }}>24 cols</span>
        </div>
      </div>

      {/* Arrow CSV → Feature Eng */}
      <ArrowWithParticles x1={152} y1={72} x2={210} y2={130} tick={tick} />

      {/* === Feature Engineering Node === */}
      <div
        className="absolute glass-card rounded-2xl p-4 w-[148px] animate-float-delayed"
        style={{ top: 110, left: 195 }}
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: "rgba(117,117,117,0.5)" }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#B0B0B0" strokeWidth="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
            </svg>
          </div>
          <span className="text-xs font-semibold" style={{ color: "#E0E0E0" }}>Feature Eng</span>
        </div>
        <div className="flex flex-wrap gap-1">
          {["Scale", "Encode", "PCA", "Select"].map((t) => (
            <span key={t} className="text-[8px] px-1.5 py-0.5 rounded" style={{ background: "rgba(82,82,82,0.5)", color: "#B0B0B0" }}>{t}</span>
          ))}
        </div>
      </div>

      {/* Arrows from Feature Eng → 3 models */}
      <ArrowWithParticles x1={270} y1={160} x2={100} y2={250} tick={tick} delay={20} />
      <ArrowWithParticles x1={270} y1={162} x2={270} y2={250} tick={tick} delay={10} />
      <ArrowWithParticles x1={270} y1={160} x2={390} y2={250} tick={tick} delay={30} />

      {/* === 3 Model Cards === */}
      {[
        { label: "Random Forest", Icon: TreePine, acc: "94.2%", left: 10 },
        { label: "Grad Boosting", Icon: Zap, acc: "95.8%", left: 195 },
        { label: "Logistic Reg", Icon: BarChart3, acc: "89.1%", left: 330 },
      ].map(({ label, Icon, acc, left }, i) => (
        <div
          key={label}
          className="absolute glass-card rounded-xl p-3 w-[130px]"
          style={{
            top: 248,
            left,
            animation: `float ${4 + i}s ease-in-out infinite`,
            animationDelay: `${i * 0.8}s`,
          }}
        >
          <div className="mb-1">
            <Icon className="text-[#B0B0B0]" size={20} />
          </div>
          <div className="text-[10px] font-semibold mb-1" style={{ color: "#E0E0E0" }}>{label}</div>
          <div className="text-xs font-bold" style={{ color: "#B0B0B0" }}>{acc}</div>
          <div className="mt-1.5 h-1 rounded-full" style={{ background: "rgba(82,82,82,0.5)" }}>
            <div
              className="h-full rounded-full"
              style={{ width: acc, background: "linear-gradient(90deg, #757575, #8C8C8C)" }}
            />
          </div>
        </div>
      ))}

      {/* Arrows → Best Model */}
      <ArrowWithParticles x1={75} y1={322} x2={200} y2={390} tick={tick} delay={5} />
      <ArrowWithParticles x1={260} y1={322} x2={252} y2={390} tick={tick} delay={15} />
      <ArrowWithParticles x1={395} y1={322} x2={300} y2={390} tick={tick} delay={25} />

      {/* === Best Model Selected === */}
      <div
        className="absolute rounded-2xl p-4 w-[200px] animate-float"
        style={{
          top: 385,
          left: 155,
          background: "linear-gradient(135deg, rgba(117,117,117,0.35) 0%, rgba(82,82,82,0.25) 100%)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(140,140,140,0.6)",
          boxShadow: "0 0 24px rgba(117,117,117,0.4), 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1)",
        }}
      >
        <div className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 rounded-full animate-pulse-glow" style={{ background: "#B0B0B0" }} />
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "#B0B0B0" }}>Best Model</span>
        </div>
        <div className="text-sm font-bold" style={{ color: "#E0E0E0" }}>Gradient Boosting</div>
        <div className="flex gap-2 mt-2">
          <span className="text-[9px] px-2 py-0.5 rounded-full" style={{ background: "rgba(117,117,117,0.5)", color: "#B0B0B0" }}>AUC 0.97</span>
          <span className="text-[9px] px-2 py-0.5 rounded-full" style={{ background: "rgba(117,117,117,0.5)", color: "#B0B0B0" }}>F1 0.958</span>
        </div>
      </div>

      {/* Arrow → API */}
      {/* <ArrowWithParticles x1={255} y1={460} x2={390} y2={400} tick={tick} delay={35} /> */}

      {/* === API Deployment === */}
      {/* <div
        className="absolute glass-card rounded-2xl p-4 w-[150px] animate-float-delayed"
        style={{ top: 340, left: 370 }}
      >
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full animate-server-pulse" style={{ background: "#B0B0B0" }} />
          <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: "#B0B0B0" }}>API Server</span>
        </div>
        <div className="text-[9px] font-mono mb-2" style={{ color: "#8C8C8C" }}>POST /v1/predict</div>
        <div className="text-[8px] px-2 py-1 rounded-lg font-mono" style={{ background: "rgba(45,45,45,0.8)", color: "#B0B0B0", border: "1px solid rgba(140,140,140,0.3)" }}>
          {`{ "score": 0.97 }`}
        </div>
        <div className="mt-2 flex items-center gap-1">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: "#B0B0B0", boxShadow: "0 0 6px rgba(176,176,176,0.8)" }} />
          <span className="text-[8px]" style={{ color: "#B0B0B0" }}>Running</span>
        </div>
      </div> */}
    </div>
  );
}

function ArrowWithParticles({ x1, y1, x2, y2, tick, delay = 0 }) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const angle = Math.atan2(dy, dx) * (180 / Math.PI);

  const particles = [0, 1, 2];

  return (
    <div
      className="absolute pointer-events-none"
      style={{ top: y1, left: x1, width: len, height: 2, transformOrigin: "0 50%", transform: `rotate(${angle}deg)` }}
    >
      {/* Line */}
      <div
        className="absolute"
        style={{
          top: 0, left: 0, right: 0, height: 1,
          background: "linear-gradient(90deg, rgba(140,140,140,0.15), rgba(140,140,140,0.4), rgba(140,140,140,0.15))",
        }}
      />
      {/* Particles */}
      {particles.map((p) => {
        const progress = ((tick + delay + p * 33) % 100) / 100;
        return (
          <div
            key={p}
            className="absolute w-1.5 h-1.5 rounded-full"
            style={{
              top: -3,
              left: `${progress * 100}%`,
              background: "#B0B0B0",
              boxShadow: "0 0 6px rgba(176,176,176,0.8)",
              opacity: Math.sin(progress * Math.PI),
            }}
          />
        );
      })}
    </div>
  );
}
