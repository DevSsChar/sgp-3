"use client";
import Link from "next/link";

const NAV_LINKS = ["Platform", "Features", "Pricing", "Docs", "Enterprise"];

export default function Navbar() {
  return (
    <nav className="relative z-50 flex items-center justify-between px-8 py-5 max-w-7xl mx-auto">
      <Link href="/" className="flex items-center gap-3">
        {/* <div 
          className="w-8 h-8 rounded-xl flex items-center justify-center" 
          style={{ background: "linear-gradient(135deg, #757575, #8C8C8C)" }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2D2D2D" strokeWidth="2.5">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div> */}
        <span className="text-lg font-bold tracking-tight" style={{ color: "#E0E0E0" }}>
          ELITE<span style={{ color: "#8C8C8C" }}>ML</span>
        </span>
      </Link>
      
      <div className="hidden md:flex items-center gap-8">
        {NAV_LINKS.map((l) => (
          <a 
            key={l} 
            href="#" 
            className="text-sm transition-colors duration-200" 
            style={{ color: "#B0B0B0" }}
            onMouseEnter={(e) => e.target.style.color = "#E0E0E0"}
            onMouseLeave={(e) => e.target.style.color = "#B0B0B0"}
          >
            {l}
          </a>
        ))}
      </div>
      
      <div className="flex items-center gap-3">
        <Link 
          href="/upload" 
          className="text-sm px-4 py-2 rounded-xl transition-all duration-200 font-medium"
          style={{ color: "#B0B0B0", border: "1px solid rgba(140,140,140,0.4)" }}
          onMouseEnter={(e) => { 
            e.currentTarget.style.borderColor = "rgba(140,140,140,0.7)"; 
            e.currentTarget.style.color = "#E0E0E0"; 
          }}
          onMouseLeave={(e) => { 
            e.currentTarget.style.borderColor = "rgba(140,140,140,0.4)"; 
            e.currentTarget.style.color = "#B0B0B0"; 
          }}
        >
          Sign In
        </Link>
        <Link 
          href="/upload" 
          className="text-sm px-5 py-2 rounded-xl font-semibold transition-all duration-200"
          style={{ background: "#757575", color: "#E0E0E0" }}
          onMouseEnter={(e) => e.currentTarget.style.background = "#8C8C8C"}
          onMouseLeave={(e) => e.currentTarget.style.background = "#757575"}
        >
          Get Started
        </Link>
      </div>
    </nav>
  );
}
