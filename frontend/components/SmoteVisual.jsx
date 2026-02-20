export default function SmoteVisual() {
  const majority = [[15,30],[25,45],[35,25],[45,50],[55,35],[60,55],[70,40],[80,28],[90,48]];
  const minority = [[20,65],[40,70],[60,72]];
  const synthetic = [[30,67],[50,68],[35,75],[55,63]];
  
  return (
    <div className="w-full h-20 relative overflow-hidden mt-3">
      <svg viewBox="0 0 120 80" className="w-full h-full">
        {majority.map(([cx,cy],i) => (
          <circle 
            key={`maj-${i}`} 
            cx={cx} 
            cy={cy} 
            r="2.5" 
            fill="rgba(140,140,140,0.5)" 
          />
        ))}
        {minority.map(([cx,cy],i) => (
          <circle 
            key={`min-${i}`} 
            cx={cx} 
            cy={cy} 
            r="2.5" 
            fill="rgba(176,176,176,0.8)" 
          />
        ))}
        {synthetic.map(([cx,cy],i) => (
          <g key={`syn-${i}`}>
            <circle 
              cx={cx} 
              cy={cy} 
              r="2.5" 
              fill="rgba(176,176,176,0.5)" 
              strokeDasharray="2,1" 
              stroke="#B0B0B0" 
              strokeWidth="0.5" 
            />
            <circle 
              cx={cx} 
              cy={cy} 
              r="5" 
              fill="none" 
              stroke="rgba(176,176,176,0.2)" 
              strokeWidth="0.5" 
            />
          </g>
        ))}
      </svg>
    </div>
  );
}
