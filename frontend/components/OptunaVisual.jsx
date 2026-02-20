export default function OptunaVisual() {
  const pts = [
    { x: 20, y: 55 }, { x: 35, y: 40 }, { x: 50, y: 65 }, { x: 60, y: 30 },
    { x: 75, y: 50 }, { x: 85, y: 18 }, { x: 95, y: 25 },
  ];
  
  return (
    <div className="w-full h-20 relative overflow-hidden mt-3">
      <svg viewBox="0 0 120 80" className="w-full h-full">
        {pts.map((p, i) => (
          <circle 
            key={i} 
            cx={p.x} 
            cy={p.y} 
            r={i === 5 ? 4 : 2}
            fill={i === 5 ? "#B0B0B0" : "rgba(140,140,140,0.4)"}
            className={i === 5 ? "animate-pulse" : ""}
          />
        ))}
        <path 
          d="M20,55 Q50,35 85,18" 
          stroke="rgba(140,140,140,0.5)" 
          strokeWidth="1" 
          fill="none" 
          strokeDasharray="3,2" 
        />
        {pts[5] && (
          <circle 
            cx={pts[5].x} 
            cy={pts[5].y} 
            r="8" 
            fill="none" 
            stroke="rgba(176,176,176,0.3)" 
            strokeWidth="1" 
            className="animate-ping" 
          />
        )}
      </svg>
    </div>
  );
}
