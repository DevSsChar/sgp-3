"use client";

import { useMemo } from "react";
import { Download, Trophy, Clock3, Boxes, Layers, CheckCircle2 } from "lucide-react";
import Navbar from "@/components/Navbar";

export default function CNNResultsView({ resultData }) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const metricsEntries = useMemo(() => {
    const metrics = resultData?.metrics || {};
    return Object.entries(metrics).filter(([key]) => key !== "class_names");
  }, [resultData]);

  const classNames = resultData?.metrics?.class_names || [];

  const bestModelUrl = resultData?.download_urls?.best_model ? `${apiUrl}${resultData.download_urls.best_model}` : null;
  const metricsUrl = resultData?.download_urls?.metrics ? `${apiUrl}${resultData.download_urls.metrics}` : null;

  const openDownload = (url) => {
    if (url) window.open(url, "_blank");
  };

  return (
    <div className="min-h-screen bg-[#000000] text-[#E0E0E0]" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "linear-gradient(rgba(117, 117, 117, 0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(117, 117, 117, 0.06) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-[#757575]/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-[#757575]/5 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10">
        <Navbar />
      </div>

      <main className="relative z-10 max-w-6xl mx-auto px-8 py-8">
        <div className="flex items-center justify-between mb-8 pb-6 border-b border-[#8C8C8C]/20">
          <div className="flex items-center gap-4">
            <div className="h-1 w-12 bg-[#757575] rounded-full" />
            <h1 className="text-sm font-bold tracking-[0.3em] uppercase text-[#8C8C8C]">CNN Training Results</h1>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => openDownload(bestModelUrl)}
              disabled={!bestModelUrl}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#757575] text-[#E0E0E0] font-bold text-sm border border-[#8C8C8C]/40 disabled:opacity-50"
            >
              <Download size={16} />
              Best Model
            </button>
            <button
              onClick={() => openDownload(metricsUrl)}
              disabled={!metricsUrl}
              className="flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#525252] text-[#E0E0E0] font-bold text-sm border border-[#8C8C8C]/40 disabled:opacity-50"
            >
              <Download size={16} />
              Metrics JSON
            </button>
          </div>
        </div>

        <div
          className="rounded-xl p-8 flex flex-col lg:flex-row gap-12 items-center relative overflow-hidden mb-8"
          style={{
            background: "linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)",
            backdropFilter: "blur(20px)",
            border: "1px solid rgba(140, 140, 140, 0.3)",
            boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
          }}
        >
          <div className="absolute top-0 right-0 w-64 h-64 bg-[#757575]/5 blur-[80px] rounded-full -translate-y-1/2 translate-x-1/2" />

          <div className="flex-1 flex flex-col gap-8 relative z-10">
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold uppercase tracking-widest">
                Completed
              </span>
              <span className="text-[#8C8C8C] text-xs">Job: {resultData?.job_id || "N/A"}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">Best Architecture</p>
                <p className="text-4xl font-bold text-white mt-2">{resultData?.best_architecture || "N/A"}</p>
              </div>
              <div>
                <p className="text-[#8C8C8C] text-xs font-bold uppercase tracking-[0.2em]">Best Accuracy</p>
                <p className="text-4xl font-bold text-white mt-2">
                  {typeof resultData?.best_accuracy === "number"
                    ? `${(resultData.best_accuracy * 100).toFixed(2)}%`
                    : "N/A"}
                </p>
              </div>
            </div>

            <div className="flex gap-6 flex-wrap text-[#8C8C8C] text-xs">
              <div className="flex items-center gap-2"><Trophy size={14} /> Auto-selected best CNN</div>
              <div className="flex items-center gap-2"><CheckCircle2 size={14} /> Ready for inference</div>
            </div>
          </div>

          <div className="w-full lg:w-px h-px lg:h-32 bg-[#8C8C8C]/20" />

          <div className="flex flex-col gap-5 min-w-[220px] text-sm">
            <div className="flex items-center justify-between gap-4">
              <span className="text-[#8C8C8C] flex items-center gap-2"><Boxes size={14} /> Dataset Size</span>
              <span className="text-white font-bold">{resultData?.metrics?.dataset_size ?? "N/A"}</span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-[#8C8C8C] flex items-center gap-2"><Layers size={14} /> Classes</span>
              <span className="text-white font-bold">{resultData?.metrics?.num_classes ?? "N/A"}</span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-[#8C8C8C] flex items-center gap-2"><Clock3 size={14} /> Epochs / Batch</span>
              <span className="text-white font-bold">{resultData?.metrics?.epochs ?? "N/A"} / {resultData?.metrics?.batch_size ?? "N/A"}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div
            className="rounded-xl p-6"
            style={{
              background: "linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(140, 140, 140, 0.3)",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
            }}
          >
            <h3 className="font-bold text-white mb-4">Metrics</h3>
            <div className="space-y-3">
              {metricsEntries.length > 0 ? (
                metricsEntries.map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center py-2 border-b border-[#8C8C8C]/10 text-sm">
                    <span className="text-[#8C8C8C]">{key}</span>
                    <span className="text-white font-mono">{typeof value === "number" ? value.toString() : String(value)}</span>
                  </div>
                ))
              ) : (
                <p className="text-[#8C8C8C] text-sm">No metrics available.</p>
              )}
            </div>
          </div>

          <div
            className="rounded-xl p-6"
            style={{
              background: "linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)",
              backdropFilter: "blur(20px)",
              border: "1px solid rgba(140, 140, 140, 0.3)",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.3)",
            }}
          >
            <h3 className="font-bold text-white mb-4">Class Names</h3>
            {classNames.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {classNames.map((name) => (
                  <span key={name} className="px-3 py-1 rounded-full text-xs"
                    style={{
                      background: "rgba(117, 117, 117, 0.25)",
                      border: "1px solid rgba(140, 140, 140, 0.3)",
                    }}
                  >
                    {name}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-[#8C8C8C] text-sm">No class names returned by backend.</p>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
