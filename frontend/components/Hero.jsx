"use client";
import Link from "next/link";
import PipelineViz from "@/components/PipelineViz";

const STATS = [
  { value: "10x", label: "Faster than manual ML" },
  { value: "97%", label: "Average AUC on benchmarks" },
  { value: "500+", label: "Optuna trials per run" },
  { value: "< 5min", label: "CSV to deployed API" },
];

export default function Hero() {
  return (
    <section className="relative z-10 max-w-7xl mx-auto px-8 pt-16 pb-24">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-start">
        {/* Left */}
        <div>
          <div 
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-6 text-xs font-medium"
            style={{ background: "rgba(117,117,117,0.2)", border: "1px solid rgba(140,140,140,0.3)", color: "#B0B0B0" }}
          >
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "#B0B0B0" }} />
            Enterprise AutoML Platform · v3.1
          </div>
          
          <h1 
            className="font-bold leading-tight mb-6 text-glow" 
            style={{ fontSize: "clamp(40px,5vw,64px)", color: "#E0E0E0", letterSpacing: "-0.02em" }}
          >
            Turn CSV Data into<br />
            <span style={{ color: "#B0B0B0" }}>Production-Ready AI</span><br />
            — Instantly.
          </h1>
          
          <p 
            className="text-base mb-10 leading-relaxed max-w-lg" 
            style={{ color: "#8C8C8C", letterSpacing: "0.01em" }}
          >
            Automated model selection. Optuna tuning. Drift detection. Deployment-ready model.
          </p>
          
          <div className="flex flex-wrap gap-4 mb-12">
            <Link 
              href="/upload" 
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-250"
              style={{ background: "#757575", color: "#E0E0E0", boxShadow: "0 4px 20px rgba(117,117,117,0.4)" }}
              onMouseEnter={(e) => { 
                e.currentTarget.style.background = "#8C8C8C"; 
                e.currentTarget.style.boxShadow = "0 8px 32px rgba(140,140,140,0.5)"; 
              }}
              onMouseLeave={(e) => { 
                e.currentTarget.style.background = "#757575"; 
                e.currentTarget.style.boxShadow = "0 4px 20px rgba(117,117,117,0.4)"; 
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="13 17 18 12 13 7"/><path d="M6 12h12"/>
              </svg>
              Launch Platform
            </Link>
            <a 
              href="#features" 
              className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl font-semibold text-sm transition-all duration-250"
              style={{ color: "#B0B0B0", border: "1px solid rgba(140,140,140,0.4)", background: "transparent" }}
              onMouseEnter={(e) => { 
                e.currentTarget.style.borderColor = "rgba(140,140,140,0.7)"; 
                e.currentTarget.style.color = "#E0E0E0"; 
              }}
              onMouseLeave={(e) => { 
                e.currentTarget.style.borderColor = "rgba(140,140,140,0.4)"; 
                e.currentTarget.style.color = "#B0B0B0"; 
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>
              </svg>
              Watch Demo
            </a>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">
            {STATS.map(({ value, label }) => (
              <div key={label} className="glass-card rounded-2xl p-4 relative overflow-hidden">
                <div 
                  className="absolute top-0 left-0 right-0 h-px" 
                  style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)" }} 
                />
                <div className="text-2xl font-bold mb-0.5" style={{ color: "#E0E0E0" }}>{value}</div>
                <div className="text-xs" style={{ color: "#8C8C8C" }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right — Pipeline Visualization */}
        <div className="ml-auto w-full max-w-lg -mt-8">
          <PipelineViz />
        </div>
      </div>
    </section>
  );
}
