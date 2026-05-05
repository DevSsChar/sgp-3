"use client";
import { Star, Download } from 'lucide-react';

// Reusable glass card style
const glassCardStyle = {
  background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
  backdropFilter: 'blur(20px)',
  border: '1px solid rgba(140, 140, 140, 0.3)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
};

export default function ModelDetailView({ modelData }) {
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
      <div className="glass-card rounded-xl p-8 flex flex-col lg:flex-row gap-12 items-center relative overflow-hidden mb-8" style={glassCardStyle}>
        <div className="absolute top-0 right-0 w-64 h-64 bg-[#757575]/5 blur-[80px] rounded-full -translate-y-1/2 translate-x-1/2"></div>
        
        <div className="flex-1 flex flex-col gap-8 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Test Accuracy */}
            <div className="flex flex-col gap-1">
              <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">Test Accuracy</p>
              <span className="text-7xl font-bold text-white" style={{ textShadow: '0 0 20px rgba(255, 255, 255, 0.4)' }}>
                {test_score.toFixed(4)}
              </span>
            </div>

            {/* CV Mean Score */}
            <div className="flex flex-col gap-1">
              <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">CV Mean Score</p>
              <span className="text-5xl font-bold text-white/90">{cv_score_mean.toFixed(4)}</span>
              <p className="text-[#B0B0B0] text-xs mt-1">Std Dev: ±{cv_score_std.toFixed(4)}</p>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-[#757575] via-[#8C8C8C] to-white shadow-[0_0_15px_rgba(255,255,255,0.3)]"
              style={{ width: `${test_score * 100}%` }}
            />
          </div>

          {/* Metadata Badges */}
          <div className="flex gap-6 flex-wrap text-[#8C8C8C] text-xs">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="font-medium">Production Ready</span>
            </div>
            <div className="flex items-center gap-2">
              <Star size={16} />
              <span className="font-medium">Final Score: {final_score.toFixed(4)}</span>
            </div>
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span className="font-medium">{feature_count} Features • {train_samples} Samples</span>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="w-full lg:w-px h-px lg:h-32 bg-[#8C8C8C]/20" />

        {/* Performance Metrics */}
        <div className="flex flex-col gap-6 min-w-[200px]">
          <MetricDisplay label="Latency (ms)" value={latency_ms.toFixed(3)} unit="ms" />
          <MetricDisplay label="Model Size" value={size_mb.toFixed(1)} unit="MB" />
          <div className="flex flex-col gap-2">
            <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Task Type</p>
            <p className="text-lg font-bold text-white capitalize">{task}</p>
          </div>
        </div>
      </div>

      {/* Specifications Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Hyperparameters */}
        <InfoCard 
          title="Hyperparameters"
          icon={
            <svg className="w-5 h-5 text-[#757575]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          }
          data={hyperparameters}
        />

        {/* Model Information */}
        <InfoCard 
          title="Model Information"
          icon={
            <svg className="w-5 h-5 text-[#757575]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          data={{
            'Model Type': model_name,
            'Dataset Hash': model_id.split('_')[1],
            'Feature Count': feature_count,
            'Training Samples': train_samples
          }}
        />
      </div>
    </div>
  );
}

// Reusable Metric Display Component
function MetricDisplay({ label, value, unit }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">{label}</p>
      <p className="text-2xl font-bold text-white">
        {value}
        <span className="text-sm text-[#8C8C8C] ml-1">{unit}</span>
      </p>
    </div>
  );
}

// Reusable Info Card Component
function InfoCard({ title, icon, data }) {
  return (
    <div className="glass-card rounded-xl p-6 flex flex-col gap-6" style={glassCardStyle}>
      <h3 className="font-bold text-white flex items-center gap-2">
        {icon}
        {title}
      </h3>
      <div className="space-y-4">
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
            <span className="text-sm text-[#8C8C8C]">{key}</span>
            <span className="text-sm font-mono font-bold text-white">
              {typeof value === 'number' ? value.toFixed(4) : value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
