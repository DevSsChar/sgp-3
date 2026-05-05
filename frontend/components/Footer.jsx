export default function Footer() {
  const footerLinks = ["Privacy", "Terms", "Security", "Status"];

  return (
    <footer 
      className="relative z-10 border-t px-8 py-10" 
      style={{ borderColor: "rgba(140,140,140,0.2)" }}
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {/* <div 
            className="w-7 h-7 rounded-lg flex items-center justify-center" 
            style={{ background: "linear-gradient(135deg, #757575, #8C8C8C)" }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#2D2D2D" strokeWidth="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
          </div> */}
          <span className="text-sm font-semibold" style={{ color: "#B0B0B0" }}>ELITEML</span>
        </div>
        
        <p className="text-xs" style={{ color: "#757575" }}>
          © 2026 ELITE AutoML. Enterprise AI Infrastructure.
        </p>
        
        <div className="flex gap-6">
          {footerLinks.map((l) => (
            <a 
              key={l} 
              href="#" 
              className="text-xs transition-colors duration-200"
              style={{ color: "#757575" }}
              onMouseEnter={(e) => e.target.style.color = "#B0B0B0"}
              onMouseLeave={(e) => e.target.style.color = "#757575"}
            >
              {l}
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
