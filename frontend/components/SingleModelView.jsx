"use client";
import { Download, Star, CheckCircle, BarChart3, Settings as SettingsIcon, Zap } from 'lucide-react';
import { GLASS_CARD_STYLE } from '@/lib/constants';

export default function SingleModelView({ modelData }) {
  const {
    model_id,
    model_name,
    task,
    cv_score_mean,
    cv_score_std,
    test_score,
    latency_ms,
    size_mb,
    final_score,
    hyperparameters,
    timestamp,
    feature_count,
    train_samples,
    pickle_download
  } = modelData;

  const formattedDate = new Date(timestamp).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  const handleDownload = () => {
    if (pickle_download) {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      window.open(`${apiUrl}${pickle_download}`, '_blank');
    }
  };

  return (
    <div className="w-full">
      {/* Top Header Bar */}
      <div className="flex items-center justify-between mb-8 pb-6 border-b border-[#8C8C8C]/20">
        <div className="flex items-center gap-4">
          <div className="h-1 w-12 bg-[#757575] rounded-full"></div>
          <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-[#8C8C8C]">Model Details</h1>
        </div>
        <button
          onClick={handleDownload}
          className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-[#757575] text-[#E0E0E0] font-bold text-sm border border-[#8C8C8C]/40 hover:bg-[#8C8C8C] transition-all shadow-xl shadow-[#757575]/20"
        >
          <Download size={18} />
          Download .pkl
        </button>
      </div>

      {/* Header Info */}
      <div className="flex flex-col gap-2 mb-8">
        <div className="flex items-center gap-3 flex-wrap">
          <span className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold uppercase tracking-widest">
            Optimized Model
          </span>
          <span className="text-[#8C8C8C] text-xs">•</span>
          <span className="text-[#8C8C8C] text-xs font-medium uppercase tracking-wider">
            Run ID: {model_id.split('_').pop()}
          </span>
          <span className="text-[#8C8C8C] text-xs">•</span>
          <span className="text-[#8C8C8C] text-xs font-medium">{formattedDate}</span>
        </div>
        <h2 className="text-4xl font-bold tracking-tight text-white">
          {model_name}_{model_id.split('_').pop()}
        </h2>
      </div>

      {/* Main Score Card */}
      <div className="glass-card rounded-xl p-8 flex flex-col lg:flex-row gap-12 items-center relative overflow-hidden mb-8" style={GLASS_CARD_STYLE}>
        <div className="absolute top-0 right-0 w-64 h-64 bg-[#757575]/5 blur-[80px] rounded-full -translate-y-1/2 translate-x-1/2"></div>
        
        <div className="flex-1 flex flex-col gap-8 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex flex-col gap-1">
              <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">Test Accuracy</p>
              <span className="text-7xl font-bold text-white" style={{ textShadow: '0 0 20px rgba(255, 255, 255, 0.4)' }}>
                {test_score.toFixed(4)}
              </span>
            </div>

            <div className="flex flex-col gap-1">
              <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">CV Mean Score</p>
              <span className="text-5xl font-bold text-white/90">{cv_score_mean.toFixed(4)}</span>
              <p className="text-[#B0B0B0] text-xs mt-1">Std Dev: ±{cv_score_std.toFixed(4)}</p>
            </div>
          </div>

          <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-[#757575] via-[#8C8C8C] to-white shadow-[0_0_15px_rgba(255,255,255,0.3)]"
              style={{ width: `${test_score * 100}%` }}
            ></div>
          </div>

          <div className="flex gap-6 flex-wrap">
            <div className="flex items-center gap-2 text-[#8C8C8C]">
              <CheckCircle size={16} />
              <span className="text-xs font-medium">Production Ready</span>
            </div>
            <div className="flex items-center gap-2 text-[#8C8C8C]">
              <Star size={16} />
              <span className="text-xs font-medium">Final Score: {final_score.toFixed(4)}</span>
            </div>
            <div className="flex items-center gap-2 text-[#8C8C8C]">
              <BarChart3 size={16} />
              <span className="text-xs font-medium">{feature_count} Features • {train_samples} Samples</span>
            </div>
          </div>
        </div>

        <div className="w-full lg:w-px h-px lg:h-32 bg-[#8C8C8C]/20"></div>

        <div className="flex flex-col gap-6 min-w-[200px]">
          <div className="flex flex-col gap-2">
            <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Latency (ms)</p>
            <p className="text-2xl font-bold text-white">
              {latency_ms.toFixed(3)}<span className="text-sm text-[#8C8C8C] ml-1">ms</span>
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Model Size</p>
            <p className="text-2xl font-bold text-white">
              {size_mb.toFixed(1)}<span className="text-sm text-[#8C8C8C] ml-1">MB</span>
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Task Type</p>
            <p className="text-lg font-bold text-white capitalize">{task}</p>
          </div>
        </div>
      </div>

      {/* Specifications Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card rounded-xl p-6 flex flex-col gap-6" style={GLASS_CARD_STYLE}>
          <h3 className="font-bold text-white flex items-center gap-2">
            <SettingsIcon size={20} className="text-[#757575]" />
            Hyperparameters
          </h3>
          <div className="space-y-4">
            {Object.entries(hyperparameters).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
                <span className="text-sm text-[#8C8C8C]">{key}</span>
                <span className="text-sm font-mono font-bold text-white">
                  {typeof value === 'number' ? value.toFixed(4) : value}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-card rounded-xl p-6 flex flex-col gap-6" style={GLASS_CARD_STYLE}>
          <h3 className="font-bold text-white flex items-center gap-2">
            <Zap size={20} className="text-[#757575]" />
            Model Information
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
              <span className="text-sm text-[#8C8C8C]">Model Type</span>
              <span className="text-sm font-mono font-bold text-white">{model_name}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
              <span className="text-sm text-[#8C8C8C]">Dataset Hash</span>
              <span className="text-sm font-mono font-bold text-white">{model_id.split('_')[1]}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
              <span className="text-sm text-[#8C8C8C]">Feature Count</span>
              <span className="text-sm font-mono font-bold text-white">{feature_count}</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
              <span className="text-sm text-[#8C8C8C]">Training Samples</span>
              <span className="text-sm font-mono font-bold text-white">{train_samples}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
