"use client";
import Link from "next/link";

export default function CTA() {
  return (
    <section className="relative z-10 max-w-4xl mx-auto px-8 py-24 text-center">
      <div className="glass-card rounded-3xl p-12 relative overflow-hidden">
        <div 
          className="absolute top-0 left-0 right-0 h-px" 
          style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent)" }} 
        />
        <div 
          className="absolute inset-0 rounded-3xl pointer-events-none" 
          style={{ background: "radial-gradient(ellipse at 50% 0%, rgba(117,117,117,0.2) 0%, transparent 70%)" }} 
        />
        
        <h2 
          className="font-bold mb-4 relative" 
          style={{ fontSize: "clamp(28px,4vw,40px)", color: "#E0E0E0" }}
        >
          Start Automating Your ML Workflow
        </h2>
        <p className="text-base mb-8 relative" style={{ color: "#8C8C8C" }}>
          Join 2,000+ ML engineers shipping production models faster with ELITE AutoML.
        </p>
        
        <div className="flex flex-wrap gap-4 justify-center relative">
          <Link 
            href="/dashboard" 
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-sm transition-all duration-250"
            style={{ background: "#757575", color: "#E0E0E0", boxShadow: "0 4px 24px rgba(117,117,117,0.5)" }}
            onMouseEnter={(e) => { e.currentTarget.style.background = "#8C8C8C"; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "#757575"; }}
          >
            Start Free Trial
          </Link>
          <Link 
            href="/dashboard" 
            className="inline-flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-sm transition-all duration-250"
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
            View Documentation
          </Link>
        </div>
      </div>
    </section>
  );
}
