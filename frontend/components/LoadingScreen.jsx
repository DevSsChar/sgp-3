"use client";
import { Pause, Terminal, Settings, Layers, Activity, CheckCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import Navbar from './Navbar';

export default function LoadingScreen({ 
  progress = 68.4,
  modelName = "Random Forest",
  currentStep = "Hyperparameter Tuning",
  stepNumber = 4,
  totalSteps = 6,
  accuracy = 94.2,
  validationLoss = 0.128,
  gpuUtilization = 88,
  memoryBandwidth = 92,
  elapsedTime = "00:42:15",
  estimatedTime = "~00:15:00",
  features = "1,248",
  isCompleted = false,
  onPause,
  onViewLogs,
  onSeeModel
}) {
  const [pulseOpacity, setPulseOpacity] = useState(0.4);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulseOpacity(prev => prev === 0.4 ? 0.7 : 0.4);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 z-50 min-h-screen overflow-hidden" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
      {/* Cinematic Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-[#2D2D2D] via-[#000000] to-[#000000]">
        <div 
          className="absolute inset-0" 
          style={{
            backgroundImage: 'radial-gradient(rgba(117, 117, 117, 0.1) 1px, transparent 1px)',
            backgroundSize: '40px 40px'
          }}
        />
        <div 
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E\")"
          }}
        />
      </div>
      
    <Navbar/>
      {/* Header Navigation */}
      {/* <nav className="relative z-50 flex justify-between items-center px-8 py-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#757575] rounded-lg flex items-center justify-center">
            <Layers className="text-white" size={20} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">
              ELITE <span className="text-[#757575]">AutoML</span>
            </h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-[#8C8C8C]">
              Enterprise AI Infrastructure
            </p>
          </div>
        </div>
        <div className="flex gap-6 items-center">
          <div 
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background: 'rgba(117, 117, 117, 0.2)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(140, 140, 140, 0.3)'
            }}
          >
            <span className="w-2 h-2 rounded-full bg-[#B0B0B0]"></span>
            <span className="text-[#E0E0E0]">GPU Cluster Active</span>
          </div>
          <button className="text-[#8C8C8C] hover:text-white transition-colors">
            <Settings size={20} />
          </button>
        </div>
      </nav> */}

      {/* Main Training Arena */}
      <main className="relative z-10 flex flex-col items-center justify-center min-h-[calc(100vh-180px)] px-12 mt-15">
        <div className="relative w-full max-w-6xl flex items-center justify-between gap-8">
          
          {/* Left Metrics */}
          <div className="w-72 flex flex-col gap-6">
            <div 
              className="p-5 rounded-xl border-l-4"
              style={{
                background: 'rgba(117, 117, 117, 0.2)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(140, 140, 140, 0.3)',
                borderLeft: '4px solid #757575'
              }}
            >
              <div className="flex justify-between items-start mb-4">
                <span className="text-xs font-semibold uppercase tracking-widest text-[#8C8C8C]">
                  Accuracy
                </span>
                <span className="text-[#757575] text-sm font-bold">{accuracy}%</span>
              </div>
              <div className="h-16 flex items-end gap-1">
                {[4, 6, 8, 10, 12, 14, 16].map((height, i) => (
                  <div 
                    key={i}
                    className="flex-1 rounded-t-sm"
                    style={{
                      background: `rgba(117, 117, 117, ${0.2 + (i * 0.1)})`,
                      height: `${height * 4}px`
                    }}
                  />
                ))}
              </div>
            </div>

            <div 
              className="p-5 rounded-xl"
              style={{
                background: 'rgba(117, 117, 117, 0.2)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(140, 140, 140, 0.3)'
              }}
            >
              <span className="text-[10px] uppercase tracking-widest text-[#8C8C8C] block mb-2">
                GPU Utilization
              </span>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xl font-bold">{gpuUtilization}%</span>
                <span className="text-[10px] text-[#8C8C8C]">72.4 GB / 80 GB</span>
              </div>
              <div className="w-full h-1 bg-[#2D2D2D] rounded-full overflow-hidden">
                <div 
                  className="h-full bg-[#B0B0B0] transition-all duration-500"
                  style={{ width: `${gpuUtilization}%` }}
                />
              </div>
            </div>
          </div>

          {/* Central Visual Core */}
          <div className="relative flex items-center justify-center flex-1" style={{ zIndex: 1 }}>
            {/* Outer Ring 1 */}
            <div 
              className="absolute w-[500px] h-[500px] flex items-center justify-center rounded-full"
              style={{
                border: '2px dashed rgba(140, 140, 140, 0.2)',
                animation: 'spin 20s linear infinite'
              }}
            >
              <div 
                className="absolute top-0 px-3 py-1 rounded-full text-[10px] font-mono"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(117, 117, 117, 0.4)',
                  color: '#757575'
                }}
              >
                LOSS: 0.04231
              </div>
              <div 
                className="absolute bottom-10 right-10 px-3 py-1 rounded-full text-[10px] font-mono"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(140, 140, 140, 0.3)',
                  color: '#8C8C8C'
                }}
              >
                EPOCH: 882
              </div>
            </div>

            {/* Outer Ring 2 */}
            <div 
              className="absolute w-[400px] h-[400px] rounded-full opacity-50"
              style={{
                border: '2px dashed rgba(140, 140, 140, 0.2)',
                animation: 'spin 30s linear infinite reverse'
              }}
            >
              <div 
                className="absolute left-0 top-1/2 -translate-y-1/2 px-3 py-1 rounded-full text-[10px] font-mono"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(140, 140, 140, 0.3)',
                  color: '#8C8C8C'
                }}
              >
                X_VAL_SCALED
              </div>
            </div>

            {/* 3D Neural Sphere Core */}
            <div className="relative w-72 h-72 flex items-center justify-center">
              <div 
                className="absolute inset-0 rounded-full blur-3xl"
                style={{ 
                  background: 'rgba(117, 117, 117, 0.1)',
                  opacity: pulseOpacity,
                  transition: 'opacity 1.5s ease-in-out'
                }}
              />
              
              {/* The Sphere */}
              <div 
                className="w-64 h-64 rounded-full relative overflow-hidden flex items-center justify-center"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(117, 117, 117, 0.3)'
                }}
              >
                {/* Neural visualization */}
                <div className="relative z-10 w-full h-full flex items-center justify-center">
                  <div 
                    className="w-48 h-48 rounded-full animate-pulse"
                    style={{ border: '1px solid rgba(176, 176, 176, 0.2)' }}
                  />
                  <div 
                    className="absolute w-32 h-32 rounded-full"
                    style={{ border: '1px solid rgba(140, 140, 140, 0.3)' }}
                  />
                  <div 
                    className="absolute w-12 h-12 rounded-full blur-xl animate-pulse"
                    style={{ background: 'rgba(117, 117, 117, 0.4)' }}
                  />
                  <Activity 
                    className="absolute text-[#757575] animate-pulse" 
                    size={40}
                    style={{ opacity: pulseOpacity, transition: 'opacity 1.5s ease-in-out' }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right Metrics */}
          <div className="w-72 flex flex-col gap-6">
            <div 
              className="p-5 rounded-xl border-r-4"
              style={{
                background: 'rgba(117, 117, 117, 0.2)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(140, 140, 140, 0.3)',
                borderRight: '4px solid #8C8C8C'
              }}
            >
              <div className="flex justify-between items-start mb-4">
                <span className="text-xs font-semibold uppercase tracking-widest text-[#8C8C8C]">
                  Validation Loss
                </span>
                <span className="text-[#E0E0E0] text-sm font-bold">{validationLoss}</span>
              </div>
              <div className="h-16 flex items-end gap-1">
                {[16, 14, 12, 10, 8, 6, 4].map((height, i) => (
                  <div 
                    key={i}
                    className="flex-1 rounded-t-sm"
                    style={{
                      background: `rgba(140, 140, 140, ${0.8 - (i * 0.1)})`,
                      height: `${height * 4}px`
                    }}
                  />
                ))}
              </div>
            </div>

            <div 
              className="p-5 rounded-xl"
              style={{
                background: 'rgba(117, 117, 117, 0.2)',
                backdropFilter: 'blur(12px)',
                border: '1px solid rgba(140, 140, 140, 0.3)'
              }}
            >
              <span className="text-[10px] uppercase tracking-widest text-[#8C8C8C] block mb-2">
                Memory Bandwidth
              </span>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xl font-bold">4.2 TB/s</span>
                <span className="text-[10px] text-[#8C8C8C]">{memoryBandwidth}% Capacity</span>
              </div>
              <div className="w-full h-1 bg-[#2D2D2D] rounded-full overflow-hidden">
                <div 
                  className="h-full bg-[#757575] transition-all duration-500"
                  style={{ width: `${memoryBandwidth}%`, opacity: 0.6 }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Status Header & Progress */}
        <div className="w-full max-w-4xl mt-12 text-center relative" style={{ zIndex: 10 }}>
          <h2 className="text-2xl font-light text-[#E0E0E0] mb-2">
            Training Best-Fit Model: <span className="text-[#757575] font-bold">{modelName}</span>
          </h2>
          {/* <p className="text-xs uppercase tracking-[0.4em] text-[#8C8C8C] mb-8">
            Optimization Step {stepNumber} of {totalSteps}: {currentStep}
          </p> */}
          
          <div className="relative">
            {/* Large Progress Bar */}
            <div 
              className="w-full h-4 rounded-full overflow-hidden"
              style={{
                background: 'rgba(45, 45, 45, 0.5)',
                border: '1px solid rgba(140, 140, 140, 0.3)',
                backdropFilter: 'blur(12px)'
              }}
            >
              <div 
                className="h-full relative rounded-full transition-all duration-500"
                style={{
                  width: `${progress}%`,
                  background: 'linear-gradient(90deg, #757575 0%, #8C8C8C 50%, #757575 100%)',
                  backgroundSize: '200% 100%',
                  boxShadow: '0 0 20px rgba(117, 117, 117, 0.4)',
                  animation: 'shimmer 3s linear infinite'
                }}
              >
                <div className="absolute inset-0 bg-white/10 blur-sm"></div>
              </div>
            </div>

            {/* Progress Percentage */}
            <div 
              className="absolute -top-8 -translate-x-1/2 transition-all duration-500"
              style={{ left: `${progress}%` }}
            >
              <span 
                className="text-xs font-bold px-2 py-1 rounded"
                style={{
                  color: '#757575',
                  background: 'rgba(117, 117, 117, 0.1)',
                  border: '1px solid rgba(117, 117, 117, 0.2)'
                }}
              >
                {progress}%
              </span>
            </div>
          </div>

          {/* Secondary Info */}
          <div className="flex justify-center gap-12 mt-8 text-xs font-mono text-[#8C8C8C]">
            <div className="flex items-center gap-2">
              <span className="text-[#757575] tracking-widest uppercase">Elapsed</span>
              <span className="text-[#E0E0E0]">{elapsedTime}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[#757575] tracking-widest uppercase">Estimated</span>
              <span className="text-[#E0E0E0]">{estimatedTime}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[#757575] tracking-widest uppercase">Features</span>
              <span className="text-[#E0E0E0]">{features} Dims</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer Controls */}
      <footer className="absolute bottom-0 w-full p-8 flex justify-between items-center z-50">
        <div className="flex gap-4">
          {!isCompleted && (
            <>
              {/* <button 
                onClick={onPause}
                className="px-6 py-2 rounded-lg text-sm font-semibold hover:bg-[#3D3D3D]/50 transition-all flex items-center gap-2"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(140, 140, 140, 0.3)'
                }}
              >
                <Pause size={16} />
                Pause Training
              </button>
              <button 
                onClick={onViewLogs}
                className="px-6 py-2 rounded-lg text-sm font-semibold hover:bg-[#3D3D3D]/50 transition-all flex items-center gap-2"
                style={{
                  background: 'rgba(117, 117, 117, 0.2)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(140, 140, 140, 0.3)'
                }}
              >
                <Terminal size={16} />
                View Live Logs
              </button> */}
            </>
          )}
          {isCompleted && (
            <button 
              onClick={onSeeModel}
              className="px-8 py-3 rounded-lg text-base font-bold hover:scale-105 transition-all flex items-center gap-3 animate-pulse"
              style={{
                background: 'linear-gradient(135deg, #757575 0%, #8C8C8C 100%)',
                color: 'white',
                boxShadow: '0 0 30px rgba(117, 117, 117, 0.5), 0 10px 40px rgba(0, 0, 0, 0.3)',
                border: '1px solid rgba(140, 140, 140, 0.5)'
              }}
            >
              <CheckCircle size={20} />
              See the Model
            </button>
          )}
        </div>

        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
            <span className="text-[10px] text-[#8C8C8C] uppercase tracking-widest">
              {isCompleted ? 'Training Complete' : ''}
            </span>
            <span className="text-xs font-medium">
              {isCompleted ? '100% Success' : ''}
            </span>
          </div>
          <div 
            className="w-12 h-12 rounded-full overflow-hidden flex items-center justify-center"
            style={{
              background: 'rgba(117, 117, 117, 0.2)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(140, 140, 140, 0.3)'
            }}
          >
            {isCompleted ? (
              <CheckCircle className="text-[#B0B0B0]" size={24} />
            ) : (
              <div className="w-8 h-8 rounded-full bg-[#757575]/20" />
            )}
          </div>
        </div>
      </footer>

      {/* CSS Animations */}
      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes shimmer {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
      `}</style>
    </div>
  );
}
