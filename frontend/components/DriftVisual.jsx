export default function DriftVisual() {
  return (
    <div className="w-full h-20 relative overflow-hidden mt-3">
      <svg viewBox="0 0 120 80" className="w-full h-full">
        <path 
          d="M5,70 Q30,10 60,40 Q90,70 115,70" 
          stroke="rgba(140,140,140,0.7)" 
          strokeWidth="1.5" 
          fill="none" 
        />
        <path 
          d="M15,70 Q45,8 75,35 Q105,65 120,70" 
          stroke="rgba(176,176,176,0.5)" 
          strokeWidth="1.5" 
          fill="none" 
          strokeDasharray="4,2" 
        />
        <line 
          x1="68" 
          y1="10" 
          x2="68" 
          y2="70" 
          stroke="rgba(140,140,140,0.3)" 
          strokeWidth="1" 
          strokeDasharray="2,2" 
        />
        <text x="70" y="18" fontSize="7" fill="#8C8C8C">PSI: 0.18</text>
      </svg>
    </div>
  );
}
