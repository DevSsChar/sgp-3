"use client";
import { Download, TrendingUp, Star, Clock, HardDrive, Zap, Settings as SettingsIcon, BarChart3, ExternalLink, CheckCircle, Cpu } from 'lucide-react';
import Navbar from './Navbar';

export default function ResultsView({ modelData }) {
  // Use mock data if no model data is provided
  const mockData = {
    model_id: "GradientBoosting_b22de4e9f16e_1771603320",
    model_name: "GradientBoosting",
    task: "classification",
    cv_score_mean: 0.9963594852635949,
    cv_score_std: 0.0034026260909819847,
    test_score: 1.0,
    latency_ms: 0.006632804870605469,
    size_mb: 3,
    final_score: 0.8499993367195129,
    hyperparameters: {
      n_estimators: 106,
      learning_rate: 0.2835559696877989,
      max_depth: 10,
      min_samples_split: 3,
      subsample: 0.6014947911252734
    },
    dataset_hash: "b22de4e9f16e",
    timestamp: new Date().toISOString(),
    feature_count: 136,
    train_samples: 1097,
    pickle_download: "/models/GradientBoosting_b22de4e9f16e_1771603320/download"
  };

  const data = modelData || mockData;

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
  } = data;

  // Format timestamp
  const formattedDate = new Date(timestamp).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  // Calculate percentage change (mock for demo)
  const accuracyChange = 0.0;
  const cvChange = 0.002;

  // Handle download
  const handleDownload = async () => {
    if (pickle_download) {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      window.open(`${apiUrl}${pickle_download}`, '_blank');
    }
  };

  return (
    <div className="min-h-screen bg-[#000000] text-[#E0E0E0]" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
      {/* Background Grid Pattern */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div 
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage: 'linear-gradient(rgba(117, 117, 117, 0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(117, 117, 117, 0.06) 1px, transparent 1px)',
            backgroundSize: '48px 48px'
          }}
        />
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#757575]/10 blur-[120px] rounded-full"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-[#757575]/5 blur-[120px] rounded-full"></div>
      </div>

      {/* Navbar */}
      <div className="relative z-10">
        <Navbar />
      </div>

      {/* Main Content */}
      <main className="relative z-10 max-w-6xl mx-auto px-8 py-8">
        {/* Top Header Bar */}
        <div className="flex items-center justify-between mb-8 pb-6 border-b border-[#8C8C8C]/20">
          <div className="flex items-center gap-4">
            <div className="h-1 w-12 bg-[#757575] rounded-full"></div>
            <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-[#8C8C8C]">Results Explorer</h1>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-[#757575] text-[#E0E0E0] font-bold text-sm border border-[#8C8C8C]/40 hover:bg-[#8C8C8C] transition-all shadow-xl shadow-[#757575]/20"
            >
              <Download size={18} />
              Download .pkl
            </button>
          </div>
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
            <span className="text-[#8C8C8C] text-xs font-medium">
              {formattedDate}
            </span>
          </div>
          <h2 className="text-4xl font-bold tracking-tight text-white">
            {model_name}_{model_id.split('_').pop()}
          </h2>
        </div>

        {/* Main Score Card */}
        <div 
          className="glass-card rounded-xl p-8 flex flex-col lg:flex-row gap-12 items-center relative overflow-hidden mb-8"
          style={{
            background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(140, 140, 140, 0.3)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
          }}
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#757575]/5 blur-[80px] rounded-full -translate-y-1/2 translate-x-1/2"></div>
          
          <div className="flex-1 flex flex-col gap-8 relative z-10">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Test Accuracy */}
              <div className="flex flex-col gap-1">
                <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">Test Accuracy</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-7xl font-bold text-white" style={{ textShadow: '0 0 20px rgba(255, 255, 255, 0.4)' }}>
                    {test_score.toFixed(4)}
                  </span>
                  {accuracyChange >= 0 && (
                    <span className="text-green-400 text-sm font-bold flex items-center gap-1">
                      <TrendingUp size={14} />
                      {accuracyChange.toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>

              {/* CV Mean Score */}
              <div className="flex flex-col gap-1">
                <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">CV Mean Score</p>
                <div className="flex items-baseline gap-2">
                  <span className="text-5xl font-bold text-white/90">
                    {cv_score_mean.toFixed(4)}
                  </span>
                  {cvChange >= 0 && (
                    <span className="text-green-400 text-sm font-bold flex items-center gap-1">
                      <TrendingUp size={14} />
                      {(cvChange * 100).toFixed(3)}%
                    </span>
                  )}
                </div>
                <p className="text-[#B0B0B0] text-xs mt-1">
                  Std Dev: ±{cv_score_std.toFixed(4)}
                </p>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-[#757575] via-[#8C8C8C] to-white shadow-[0_0_15px_rgba(255,255,255,0.3)]"
                style={{ width: `${test_score * 100}%` }}
              ></div>
            </div>

            {/* Metadata Badges */}
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

          {/* Divider */}
          <div className="w-full lg:w-px h-px lg:h-32 bg-[#8C8C8C]/20"></div>

          {/* Performance Metrics */}
          <div className="flex flex-col gap-6 min-w-[200px]">
            <div className="flex flex-col gap-2">
              <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Latency (ms)</p>
              <p className="text-2xl font-bold text-white">
                {latency_ms.toFixed(3)}
                <span className="text-sm text-[#8C8C8C] ml-1">ms</span>
              </p>
            </div>
            <div className="flex flex-col gap-2">
              <p className="text-[#8C8C8C] text-[10px] font-bold uppercase tracking-widest">Model Size</p>
              <p className="text-2xl font-bold text-white">
                {size_mb.toFixed(1)}
                <span className="text-sm text-[#8C8C8C] ml-1">MB</span>
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
          {/* Hyperparameters */}
          <div 
            className="glass-card rounded-xl p-6 flex flex-col gap-6"
            style={{
              background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(140, 140, 140, 0.3)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
            }}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white flex items-center gap-2">
                <SettingsIcon size={20} className="text-[#757575]" />
                Hyperparameters
              </h3>
              <span className="text-[10px] text-[#8C8C8C] uppercase font-bold tracking-widest cursor-pointer hover:text-[#B0B0B0] transition-colors">
                View Raw
              </span>
            </div>
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

          {/* Model Information */}
          <div 
            className="glass-card rounded-xl p-6 flex flex-col gap-6"
            style={{
              background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(140, 140, 140, 0.3)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
            }}
          >
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-white flex items-center gap-2">
                <Zap size={20} className="text-[#757575]" />
                Model Information
              </h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
                <span className="text-sm text-[#8C8C8C]">Model Type</span>
                <span className="text-sm font-mono font-bold text-white">{model_name}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10">
                <span className="text-sm text-[#8C8C8C]">Dataset Hash</span>
                <span className="text-sm font-mono font-bold text-white">
                  {model_id.split('_')[1]}
                </span>
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

        {/* Footer Status Bar */}
        <footer className="mt-12 px-6 py-4 rounded-xl bg-[#757575]/5 border border-[#8C8C8C]/10 flex items-center justify-between flex-wrap gap-4">
          <div className="flex gap-6 text-[10px] font-bold text-[#8C8C8C] uppercase tracking-widest flex-wrap">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              Model Ready
            </div>
            <div className="flex items-center gap-2">
              <Cpu size={14} />
              Optimized Build
            </div>
          </div>
          <div className="text-[10px] font-bold text-[#8C8C8C]/50 uppercase tracking-widest">
            EliteML  • {new Date().getFullYear()}
          </div>
        </footer>
      </main>
    </div>
  );
}
