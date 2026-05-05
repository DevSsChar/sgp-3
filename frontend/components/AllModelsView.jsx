"use client";
import { Star, Clock, Zap, TrendingUp, BarChart3, Download } from 'lucide-react';

export default function AllModelsView({ modelsData, onSelectModel }) {
  // Sort models by final_score descending
  const sortedModels = [...modelsData].sort((a, b) => b.final_score - a.final_score);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="w-full">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div className="h-1 w-12 bg-[#757575] rounded-full"></div>
          <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-[#8C8C8C]">All Models</h1>
        </div>
        <p className="text-[#B0B0B0] text-sm">
          {modelsData.length} model{modelsData.length !== 1 ? 's' : ''} trained • Sorted by final score
        </p>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 gap-6">
        {sortedModels.map((model, index) => (
          <div
            key={model.model_id}
            className="glass-card rounded-xl p-6 hover:border-[#8C8C8C]/60 transition-all cursor-pointer group"
            style={{
              background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(140, 140, 140, 0.3)',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
            }}
            onClick={() => onSelectModel(model)}
          >
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Left: Model Info */}
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  {index === 0 && (
                    <span className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold uppercase tracking-widest">
                      Best Model
                    </span>
                  )}
                  <span className="text-[#8C8C8C] text-xs">
                    {formatTimestamp(model.timestamp)}
                  </span>
                </div>
                
                <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-[#B0B0B0] transition-colors">
                  {model.model_name}
                </h3>
                
                <p className="text-[#8C8C8C] text-xs font-mono mb-4">
                  ID: {model.model_id.split('_').pop()}
                </p>

                {/* Key Metrics Row */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-[#8C8C8C] text-[10px] uppercase tracking-wider mb-1">Test Score</p>
                    <p className="text-lg font-bold text-white">{model.test_score.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-[#8C8C8C] text-[10px] uppercase tracking-wider mb-1">CV Mean</p>
                    <p className="text-lg font-bold text-white">{model.cv_score_mean.toFixed(4)}</p>
                  </div>
                  <div>
                    <p className="text-[#8C8C8C] text-[10px] uppercase tracking-wider mb-1">Latency</p>
                    <p className="text-lg font-bold text-white">{model.latency_ms.toFixed(3)}ms</p>
                  </div>
                  <div>
                    <p className="text-[#8C8C8C] text-[10px] uppercase tracking-wider mb-1">Size</p>
                    <p className="text-lg font-bold text-white">{model.size_mb}MB</p>
                  </div>
                </div>
              </div>

              {/* Right: Final Score Badge */}
              <div className="flex flex-col items-center justify-center min-w-[150px] border-l border-[#8C8C8C]/20 pl-6">
                <p className="text-[#8C8C8C] text-[10px] uppercase tracking-widest mb-2">Final Score</p>
                <div className="text-4xl font-bold text-white" style={{ textShadow: '0 0 20px rgba(255, 255, 255, 0.3)' }}>
                  {model.final_score.toFixed(3)}
                </div>
                <div className="flex items-center gap-6 mt-4">
                  <div className="flex items-center gap-1 text-[#8C8C8C] text-xs">
                    <BarChart3 size={12} />
                    <span>{model.feature_count} features</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Hyperparameters Preview */}
            <div className="mt-6 pt-6 border-t border-[#8C8C8C]/10">
              <p className="text-[#8C8C8C] text-[10px] uppercase tracking-wider mb-3">Key Hyperparameters</p>
              <div className="flex flex-wrap gap-4">
                {Object.entries(model.hyperparameters).slice(0, 5).map(([key, value]) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="text-[#B0B0B0] text-xs">{key}:</span>
                    <span className="text-white text-xs font-mono font-bold">
                      {typeof value === 'number' ? value.toFixed(3) : value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
