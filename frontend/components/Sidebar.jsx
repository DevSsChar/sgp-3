"use client";
import { Star, Layers, Rocket } from 'lucide-react';

export default function Sidebar({ activeView, onViewChange }) {
  const menuItems = [
    {
      id: 'best',
      label: 'Best Model',
      icon: Star,
    },
    {
      id: 'all',
      label: 'All Models',
      icon: Layers,
    },
  ];

  return (
    <aside className="w-72 min-h-screen border-r border-[#8C8C8C]/20 p-6">
      {/* Header */}
      {/* <div className="flex items-center gap-3 mb-8 px-2">
        <div className="bg-[#757575] p-2 rounded-lg flex items-center justify-center">
          <Rocket className="text-white" size={20} />
        </div>
        <div>
          <h2 className="text-sm font-bold tracking-tight text-[#E0E0E0] uppercase">Navigation</h2>
          <p className="text-[10px] text-[#8C8C8C] font-medium tracking-[0.2em] uppercase">Results</p>
        </div>
      </div> */}

      {/* Navigation Menu */}
      <nav className="flex flex-col gap-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeView === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onViewChange(item.id)}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all text-left ${
                isActive
                  ? 'bg-[#757575]/40 text-white border border-[#8C8C8C]/30 shadow-lg shadow-[#757575]/20'
                  : 'text-[#8C8C8C] hover:bg-white/5'
              }`}
            >
              <Icon size={18} />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer Info (Optional) */}
      <div className="mt-auto pt-8 px-2">
        <div className="text-[10px] text-[#8C8C8C]/50 uppercase tracking-widest">
          EliteML 
        </div>
      </div>
    </aside>
  );
}
