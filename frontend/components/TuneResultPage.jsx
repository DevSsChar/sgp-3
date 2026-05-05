"use client";

import { useState } from "react";

import {
  ArrowDownRight,
  CheckCircle2,
  Copy,
  Download,
  FolderOpen,
  HardDrive,
  Play,
  TrendingDown,
} from "lucide-react";
import Navbar from "@/components/Navbar";

const ARTIFACT_PATHS = {
  modelPath: "/content/easyfinetune_output/final_adapter",
  localZip: "/content/easyfinetune_model.zip",
  driveZip: "/content/drive/MyDrive/easyfinetune_models/easyfinetune_model.zip",
};

const SAMPLE_ROWS = [
  {
    prompt: "Explain the quantum tunneling effect in the context of NAND flash memory.",
    response:
      "In NAND flash, quantum tunneling (specifically Fowler-Nordheim) allows electrons to pass through a thin dielectric barrier, enabling write and erase operations without direct conduction.",
    status: "Pass",
    confidence: "0.98",
  },
  {
    prompt: "Write a Python script to optimize CUDA kernel occupancy.",
    response:
      "Use occupancy calculators, tune block size dynamically, and profile with Nsight Compute to maximize active warps per SM while balancing register and shared memory usage.",
    status: "Pass",
    confidence: "0.93",
  },
];

export default function TuneResultPage() {
  const [selectedDestination, setSelectedDestination] = useState("local");
  const selectedDownloadPath =
    selectedDestination === "local" ? ARTIFACT_PATHS.localZip : ARTIFACT_PATHS.driveZip;

  return (
    <>
      <Navbar />

      <div className="min-h-screen relative overflow-hidden" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
        <div className="fixed inset-0 z-0">
          <div className="absolute inset-0 bg-gradient-to-br from-[#2D2D2D] via-[#000000] to-[#2D2D2D]">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "linear-gradient(to right, rgba(140, 140, 140, 0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(140, 140, 140, 0.05) 1px, transparent 1px)",
                backgroundSize: "40px 40px",
              }}
            />
          </div>
          <div
            className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] rounded-full blur-[120px]"
            style={{ background: "rgba(117, 117, 117, 0.1)" }}
          />
          <div
            className="absolute -bottom-[10%] -right-[10%] w-[30%] h-[30%] rounded-full blur-[100px]"
            style={{ background: "rgba(140, 140, 140, 0.05)" }}
          />
        </div>

        <main className="relative z-10 px-6 py-8 lg:px-12 min-h-screen">
          <div className="max-w-7xl mx-auto">
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-6">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="px-2.5 py-0.5 rounded-full bg-[#525252]/40 text-[#B0B0B0] font-bold text-[10px] uppercase tracking-widest inline-flex items-center gap-1.5">
                    <CheckCircle2 size={12} /> Training Success
                  </span>
                  <span className="text-[#8C8C8C] text-xs">Experiment ID: FT-8829-X</span>
                </div>
                <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-[#E0E0E0]">
                  Results Dashboard
                </h1>
              </div>

              <div className="flex gap-3">
                <button className="bg-[#3D3D3D] text-[#E0E0E0] px-5 py-2.5 rounded-lg hover:bg-[#525252] transition-all flex items-center gap-2 text-sm font-semibold">
                  <Download size={16} />
                  {selectedDestination === "local" ? "Download Local Zip" : "Download Drive Copy"}
                </button>
                <button className="bg-[#757575] text-[#E0E0E0] px-5 py-2.5 rounded-lg hover:bg-[#8C8C8C] transition-all flex items-center gap-2 text-sm font-semibold">
                  <Play size={16} /> Run Inference
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div
                className="md:col-span-2 p-6 rounded-xl border-l-4 relative overflow-hidden"
                style={{ background: "rgba(61,61,61,0.4)", borderColor: "#757575" }}
              >
                <p className="text-[#8C8C8C] text-xs uppercase tracking-widest mb-4">Eval Loss (Final)</p>
                <div className="flex items-baseline gap-4">
                  <span className="text-5xl font-bold tracking-tight text-[#E0E0E0]">0.1242</span>
                  <span className="text-[#B0B0B0] font-bold inline-flex items-center gap-1 text-sm">
                    <TrendingDown size={14} /> -14.2%
                  </span>
                </div>
                <div className="mt-6 flex gap-4 text-xs">
                  <div className="flex flex-col">
                    <span className="text-[#8C8C8C] uppercase tracking-wider">Base Model</span>
                    <span className="text-[#E0E0E0]">Llama-3-8B-Instruct</span>
                  </div>
                  <div className="flex flex-col border-l border-[#525252] pl-4">
                    <span className="text-[#8C8C8C] uppercase tracking-wider">Quality Score</span>
                    <span className="text-[#B0B0B0]">94.8/100</span>
                  </div>
                </div>
              </div>

              <div className="bg-[#3D3D3D]/80 p-6 rounded-xl border border-[#525252]">
                <p className="text-[#8C8C8C] text-xs uppercase tracking-widest mb-4">Total Time</p>
                <span className="text-3xl font-bold tracking-tight text-[#E0E0E0]">04:12:08</span>
                <div className="mt-4 w-full h-1.5 bg-[#2D2D2D] rounded-full overflow-hidden">
                  <div className="h-full bg-[#757575] w-full" />
                </div>
                <p className="mt-4 text-[10px] text-[#8C8C8C] uppercase tracking-wider">Avg 2.4s per step</p>
              </div>

              {/* <div className="bg-[#3D3D3D]/80 p-6 rounded-xl border border-[#525252]">
                <p className="text-[#8C8C8C] text-xs uppercase tracking-widest mb-4">Model Path</p>
                <div className="bg-[#2D2D2D] p-3 rounded text-[11px] text-[#B0B0B0] break-all mb-4">
                  {ARTIFACT_PATHS.modelPath}
                </div>
                <button className="text-[10px] text-[#B0B0B0] font-bold uppercase tracking-widest inline-flex items-center gap-1.5 hover:text-[#E0E0E0]">
                  <Copy size={12} /> Copy Artifact Paths
                </button>
              </div> */}
            </div>

            <div className="bg-[#3D3D3D]/70 rounded-xl border border-[#525252] overflow-hidden">
              <div className="px-6 py-4 border-b border-[#525252] flex justify-between items-center">
                <h3 className="font-bold text-lg text-[#E0E0E0]">Validation Prompt Samples</h3>
                <span className="text-xs text-[#8C8C8C]">N=250 Test Set</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="text-[10px] text-[#8C8C8C] uppercase tracking-widest bg-[#2D2D2D]/60">
                      <th className="px-6 py-3 font-semibold">Prompt</th>
                      <th className="px-6 py-3 font-semibold">Generated Response</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#525252]">
                    {SAMPLE_ROWS.map((row) => (
                      <tr key={row.prompt} className="hover:bg-[#2D2D2D]/40 transition-colors">
                        <td className="px-6 py-4 text-xs text-[#B0B0B0] max-w-xs align-top">{row.prompt}</td>
                        <td className="px-6 py-4 text-xs text-[#E0E0E0] align-top">
                          {row.response}
                          <span className="block mt-2 text-[10px] text-[#B0B0B0] font-bold uppercase inline-flex items-center gap-1">
                            <ArrowDownRight size={12} /> {row.status} • {row.confidence} Conf
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-8 p-6 rounded-xl border border-[#525252]" style={{ background: "rgba(61,61,61,0.35)" }}>
              <label className="block text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] mb-4">
                Output Directory & Export Destination
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  type="button"
                  onClick={() => setSelectedDestination("local")}
                  className="flex items-center gap-3 p-4 rounded-lg text-left transition-colors"
                  style={{
                    border: selectedDestination === "local" ? "1px solid #757575" : "1px solid #525252",
                    background: selectedDestination === "local" ? "#2D2D2D" : "rgba(45,45,45,0.7)",
                  }}
                >
                  <HardDrive size={18} className="text-[#B0B0B0]" />
                  <div>
                    <p className="text-xs font-bold text-[#E0E0E0]">Local Zip Archive</p>
                    <p className="text-[10px] text-[#8C8C8C] break-all">{ARTIFACT_PATHS.localZip}</p>
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedDestination("drive")}
                  className="flex items-center gap-3 p-4 rounded-lg text-left transition-colors"
                  style={{
                    border: selectedDestination === "drive" ? "1px solid #757575" : "1px solid #525252",
                    background: selectedDestination === "drive" ? "#2D2D2D" : "rgba(45,45,45,0.7)",
                  }}
                >
                  <FolderOpen size={18} className="text-[#8C8C8C]" />
                  <div>
                    <p className="text-xs font-bold text-[#E0E0E0]">Google Drive</p>
                    <p className="text-[10px] text-[#8C8C8C] break-all">{ARTIFACT_PATHS.driveZip}</p>
                  </div>
                </button>
              </div>
              <p className="mt-3 text-[10px] text-[#8C8C8C] break-all">Selected download path: {selectedDownloadPath}</p>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
