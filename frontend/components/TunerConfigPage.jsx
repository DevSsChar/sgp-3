"use client";

import { useState } from "react";

import {
  Bot,
  CheckCircle2,
  ChevronDown,
  Database,
  FolderOpen,
  Play,
  Rocket,
  PlusCircle,
  Trash2,
  Upload,
  Link as LinkIcon,
  X,
} from "lucide-react";
import Navbar from "@/components/Navbar";

const MODELS = [
  {
    name: "Llama-3-8B",
    params: "8.03B Params",
    desc: "High performance instruction-tuned model for general dialogue.",
    vram: "16GB+",
    context: "8,192",
  },
  {
    name: "Mistral-7B-v0.3",
    params: "7.24B Params",
    desc: "SOTA performance for small-scale language models.",
    vram: "12GB+",
    context: "32,768",
  },
  {
    name: "Gemma-2b-it",
    params: "2.51B Params",
    desc: "Optimized for lightweight local deployment and edge compute.",
    vram: "6GB+",
    context: "8,192",
  },
];

const PROMPTS = [
  "Explain the concept of neural pruning to a 10-year-old.",
  "Write a Python script to visualize weight distributions using Plotly.",
];

export default function TunerConfigPage() {
  const [prompts, setPrompts] = useState(PROMPTS);
  const [activeDatasetMethod, setActiveDatasetMethod] = useState(null); // 'local' | 'kaggle'
  const [uploadedFile, setUploadedFile] = useState(null);
  const [kaggleHubLink, setKaggleHubLink] = useState("");

  const handleAddPrompt = () => {
    setPrompts((prev) => [...prev, ""]);
  };

  const handlePromptChange = (index, value) => {
    setPrompts((prev) => prev.map((prompt, i) => (i === index ? value : prompt)));
  };

  const handleDeletePrompt = (index) => {
    setPrompts((prev) => {
      if (prev.length <= 1) return prev;
      return prev.filter((_, i) => i !== index);
    });
  };

  const handleDatasetFileChange = (e) => {
    const file = e.target.files?.[0] || null;
    setUploadedFile(file);
    if (file) {
      setActiveDatasetMethod("local");
      setKaggleHubLink("");
    }
  };

  const handleKaggleHubLinkChange = (value) => {
    setKaggleHubLink(value);
    if (value.trim()) {
      setActiveDatasetMethod("kaggle");
      setUploadedFile(null);
    } else if (activeDatasetMethod === "kaggle") {
      setActiveDatasetMethod(null);
    }
  };

  const clearDatasetSelection = () => {
    setActiveDatasetMethod(null);
    setUploadedFile(null);
    setKaggleHubLink("");
  };

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

            {[
              { top: "25%", left: "25%" },
              { top: "33%", right: "25%" },
              { bottom: "25%", left: "33%" },
              { bottom: "50%", right: "10%" },
            ].map((pos, i) => (
              <div
                key={i}
                className="absolute w-0.5 h-0.5 rounded-full"
                style={{
                  background: "rgba(117, 117, 117, 0.4)",
                  ...pos,
                  filter: "drop-shadow(0 0 2px rgba(117, 117, 117, 0.4))",
                }}
              />
            ))}
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

        <main className="relative z-10 min-h-screen px-6 py-8 lg:px-12">
          <div className="max-w-7xl mx-auto">
            <div className="mb-12">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-5" style={{ background: "rgba(117,117,117,0.15)", border: "1px solid rgba(140,140,140,0.25)" }}>
                <span className="w-1.5 h-1.5 rounded-full bg-[#B0B0B0]" />
                <span className="text-[10px] font-bold uppercase tracking-[0.3em] text-[#B0B0B0]">Configuration Phase</span>
              </div>
              <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
                <div>
                  <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-[#E0E0E0] mb-3">
                    Fine-Tuning Setup
                  </h1>
                  <p className="max-w-2xl text-[#8C8C8C] text-base md:text-lg leading-relaxed">
                    Define hyperparameters and configure your training dataset, model family, and prompt samples in a focused, production-ready workspace.
                  </p>
                </div>
                <div className="flex items-center gap-3 self-start lg:self-auto">
                  <div className="px-4 py-3 rounded-xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                    <p className="text-[10px] uppercase tracking-widest text-[#8C8C8C]">Run Mode</p>
                    <p className="text-sm font-bold text-[#E0E0E0]">Interactive Setup</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8 items-start">
              <section className="col-span-12 lg:col-span-7 space-y-6">
                <div className="rounded-2xl p-1" style={{ background: "linear-gradient(135deg, rgba(117,117,117,0.20), rgba(140,140,140,0.08))" }}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5 rounded-[15px] p-6 md:p-7" style={{ background: "rgba(13,13,13,0.45)" }}>
                    <div className="md:col-span-2 flex items-center justify-between gap-4">
                      <div>
                        <p className="text-[10px] uppercase tracking-[0.28em] text-[#8C8C8C] mb-1">Dataset Source</p>
                        <h2 className="text-2xl font-bold text-[#E0E0E0]">Choose one input method</h2>
                      </div>
                      <div className="text-xs text-[#8C8C8C] hidden md:block">Either upload a file or enter a KaggleHub link</div>
                    </div>

                    <div
                      className={`p-5 rounded-xl relative transition-all ${activeDatasetMethod === "kaggle" ? "opacity-35 pointer-events-none" : ""}`}
                      style={{ background: "rgba(61,61,61,0.35)", border: activeDatasetMethod === "local" ? "1px solid rgba(117,117,117,0.65)" : "1px solid rgba(140,140,140,0.22)" }}
                    >
                      {activeDatasetMethod === "local" && (
                        <button
                          type="button"
                          onClick={clearDatasetSelection}
                          className="absolute top-3 right-3 p-1.5 rounded-md text-[#8C8C8C] hover:text-[#E0E0E0]"
                          style={{ background: "rgba(255,255,255,0.06)" }}
                        >
                          <X size={14} />
                        </button>
                      )}
                      <div className="flex items-center gap-2 mb-4">
                        <Upload size={16} className="text-[#757575]" />
                        <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0]">
                          Upload Dataset File
                        </label>
                      </div>
                      <label className="group flex flex-col items-center justify-center rounded-xl border border-dashed border-[#525252] bg-[#2D2D2D]/70 px-4 py-8 text-center cursor-pointer hover:border-[#757575] transition-colors">
                        <input
                          type="file"
                          accept=".csv,.json,.parquet,.zip,.jsonl"
                          onChange={handleDatasetFileChange}
                          className="hidden"
                        />
                        <Upload size={28} className="text-[#757575] mb-3" />
                        <p className="text-sm font-semibold text-[#E0E0E0]">Drag or browse your dataset</p>
                        <p className="mt-1 text-[11px] text-[#8C8C8C]">
                          {uploadedFile ? `Selected: ${uploadedFile.name}` : "CSV, JSON, Parquet, ZIP, JSONL"}
                        </p>
                      </label>
                    </div>

                    <div
                      className={`p-5 rounded-xl relative transition-all ${activeDatasetMethod === "local" ? "opacity-35 pointer-events-none" : ""}`}
                      style={{ background: "rgba(61,61,61,0.35)", border: activeDatasetMethod === "kaggle" ? "1px solid rgba(117,117,117,0.65)" : "1px solid rgba(140,140,140,0.22)" }}
                    >
                      {activeDatasetMethod === "kaggle" && (
                        <button
                          type="button"
                          onClick={clearDatasetSelection}
                          className="absolute top-3 right-3 p-1.5 rounded-md text-[#8C8C8C] hover:text-[#E0E0E0]"
                          style={{ background: "rgba(255,255,255,0.06)" }}
                        >
                          <X size={14} />
                        </button>
                      )}
                      <div className="flex items-center gap-2 mb-4">
                        <Database size={16} className="text-[#B0B0B0]" />
                        <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0]">
                          KaggleHub Link
                        </label>
                      </div>
                      <div className="relative">
                        <LinkIcon size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#8C8C8C]" />
                        <input
                          value={kaggleHubLink}
                          onChange={(e) => handleKaggleHubLinkChange(e.target.value)}
                          placeholder="kagglehub://datasets/owner/dataset"
                          className="w-full rounded-xl bg-[#2D2D2D] text-[#E0E0E0] pl-10 pr-4 py-3 border border-[#525252] focus:outline-none focus:border-[#8C8C8C]"
                        />
                      </div>
                      <p className="mt-3 text-[11px] text-[#8C8C8C] leading-relaxed">
                        Also supports kaggle.com/datasets/owner/dataset
                      </p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <div className="p-6 rounded-2xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] mb-3">
                    Model Selector
                  </label>
                  <div className="relative">
                    <select className="w-full rounded-lg bg-[#2D2D2D] text-[#E0E0E0] px-4 py-3 border border-[#525252] appearance-none focus:outline-none focus:border-[#8C8C8C]">
                      <option>Auto-Detect Optimized</option>
                      <option>Llama-3-8B-Instruct</option>
                      <option>Mistral-7B-v0.1</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-[#8C8C8C]" size={16} />
                  </div>
                  </div>

                  <div className="p-6 rounded-2xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] mb-3">
                    Method Selector
                  </label>
                  <div className="relative">
                    <select className="w-full rounded-lg bg-[#2D2D2D] text-[#E0E0E0] px-4 py-3 border border-[#525252] appearance-none focus:outline-none focus:border-[#8C8C8C]">
                      <option>LoRA (Efficient)</option>
                      <option>QLoRA (4-bit)</option>
                      <option>Full Fine-Tuning</option>
                    </select>
                    <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-[#8C8C8C]" size={16} />
                  </div>
                  </div>

                  <div className="p-6 rounded-2xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                  <label className="block text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] mb-3">
                    Number of Epochs
                  </label>
                  <input
                    type="number"
                    min={1}
                    defaultValue={3}
                    className="w-full rounded-lg bg-[#2D2D2D] text-[#E0E0E0] px-4 py-3 border border-[#525252] focus:outline-none focus:border-[#8C8C8C]"
                  />
                  </div>

                  {/* <div
                    className="p-6 rounded-2xl flex items-center justify-between"
                    style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}
                  >
                  <div>
                    <label className="block text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0]">
                      Validate Only
                    </label>
                    <p className="text-[11px] text-[#8C8C8C]">Dry run without saving weights</p>
                  </div>
                  <button
                    className="h-6 w-11 rounded-full p-1"
                    style={{ background: "#525252" }}
                    aria-label="validate toggle"
                  >
                    <span className="block h-4 w-4 rounded-full bg-[#E0E0E0]" />
                  </button>
                  </div> */}
                </div>

                <div className="p-7 rounded-2xl relative overflow-hidden" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                  <div className="flex items-center justify-between mb-5">
                    <div>
                      <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] block mb-1">
                        Test Prompts Editor
                      </label>
                      <p className="text-[11px] text-[#8C8C8C]">Optional prompts for prompt-based validation</p>
                    </div>
                    <button
                      type="button"
                      onClick={handleAddPrompt}
                      className="text-xs font-bold text-[#B0B0B0] hover:text-[#E0E0E0] inline-flex items-center gap-1"
                    >
                      <PlusCircle size={14} /> Add Prompt
                    </button>
                  </div>
                  <div className="space-y-3">
                    {prompts.map((prompt, index) => (
                      <div
                        key={`prompt-${index}`}
                        className="group relative rounded-xl border border-[#525252] bg-[#2D2D2D]/60 p-4"
                      >
                        <p className="text-xs mb-1" style={{ color: "#8C8C8C" }}>
                          PROMPT_{String(index + 1).padStart(2, "0")}
                        </p>
                        <textarea
                          value={prompt}
                          onChange={(e) => handlePromptChange(index, e.target.value)}
                          placeholder="Type your test prompt..."
                          className="w-full bg-transparent text-sm text-[#E0E0E0] resize-none h-12 focus:outline-none"
                        />
                        <button
                          type="button"
                          onClick={() => handleDeletePrompt(index)}
                          disabled={prompts.length <= 1}
                          className="absolute top-4 right-4 text-[#8C8C8C] hover:text-[#E0E0E0] opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

                {/* <aside className="col-span-12 lg:col-span-5 space-y-6">
                    <div className="rounded-2xl p-1" style={{ background: "linear-gradient(135deg, rgba(117,117,117,0.18), rgba(140,140,140,0.08))" }}>
                    <div className="p-6 rounded-[15px]" style={{ background: "rgba(13,13,13,0.45)", border: "1px solid rgba(140,140,140,0.18)" }}>
                        <div className="flex items-center justify-between mb-6">
                        <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0]">
                            Hardware Status
                        </label>
                        <span className="inline-flex items-center gap-1 text-xs text-[#B0B0B0]">
                            <span className="w-2 h-2 rounded-full bg-[#B0B0B0] animate-pulse" /> Ready
                        </span>
                        </div>
                        <div className="grid grid-cols-2 gap-6">
                        <div>
                            <p className="text-[10px] text-[#8C8C8C] mb-1">GPU TYPE</p>
                            <p className="text-xl font-bold text-[#E0E0E0]">Discrete</p>
                        </div>
                        <div>
                            <p className="text-[10px] text-[#8C8C8C] mb-1">VRAM CAPACITY</p>
                            <p className="text-xl font-bold text-[#B0B0B0]">24 GB</p>
                        </div>
                        <div className="col-span-2">
                            <p className="text-[10px] text-[#8C8C8C] mb-1">IDENTIFIER</p>
                            <p className="text-sm text-[#E0E0E0]">NVIDIA GeForce RTX 4090</p>
                        </div>
                        </div>
                    </div>
                    </div>

                    <div className="p-6 rounded-2xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] block mb-4">
                        Recommended Weights
                    </label>
                    <div className="space-y-3 max-h-[320px] overflow-y-auto pr-1">
                        {MODELS.map((model) => (
                        <div key={model.name} className="p-4 rounded-xl bg-[#2D2D2D] border border-[#525252] hover:border-[#8C8C8C] transition-colors cursor-pointer">
                            <div className="flex justify-between items-start mb-2 gap-3">
                            <h4 className="text-sm font-bold text-[#E0E0E0]">{model.name}</h4>
                            <span className="text-[10px] px-2 py-0.5 rounded bg-[#525252] text-[#B0B0B0] font-bold whitespace-nowrap">
                                {model.params}
                            </span>
                            </div>
                            <p className="text-xs text-[#8C8C8C] mb-3">{model.desc}</p>
                            <div className="flex gap-4 text-[10px] text-[#B0B0B0]">
                            <span>VRAM: {model.vram}</span>
                            <span>Ctx: {model.context}</span>
                            </div>
                        </div>
                        ))}
                    </div>
                    </div>

                    <div className="p-6 rounded-2xl" style={{ background: "rgba(61,61,61,0.35)", border: "1px solid rgba(140,140,140,0.22)" }}>
                    <label className="text-[10px] font-bold uppercase tracking-widest text-[#B0B0B0] block mb-4">
                        Pre-Flight Analytics
                    </label>
                    <div className="flex items-center gap-4">
                        <div className="h-16 w-16 rounded-full border-4 border-[#525252] border-t-[#B0B0B0] flex items-center justify-center text-xs font-bold text-[#E0E0E0]">
                        85%
                        </div>
                        <div>
                        <p className="text-sm font-bold text-[#E0E0E0] mb-1">Dataset Health: Optimal</p>
                        <p className="text-xs text-[#8C8C8C]">
                            Detected 4,102 samples. Token distribution within range and sequence length capped at 512.
                        </p>
                        </div>
                    </div>
                    </div>
                </aside> */}
            </div>

            <div className="mt-10 p-6 rounded-2xl border border-[#525252] flex flex-col md:flex-row md:items-center md:justify-between gap-5" style={{ background: "rgba(61,61,61,0.35)" }}>
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-[#525252] flex items-center justify-center">
                  <Rocket className="text-[#B0B0B0]" size={18} />
                </div>
                <div>
                  <p className="text-sm font-bold text-[#E0E0E0]">Ready for Initialization</p>
                  <p className="text-xs text-[#8C8C8C]">All parameters validated against hardware constraints.</p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button className="px-6 py-2.5 text-sm font-bold rounded-lg bg-[#757575] text-[#E0E0E0] hover:bg-[#8C8C8C] inline-flex items-center gap-2">
                  <Play size={14} /> Start Fine-Tuning
                </button>
                <button
                  disabled
                  className="px-5 py-2.5 text-sm font-bold rounded-lg bg-[#2D2D2D] text-[#8C8C8C] inline-flex items-center gap-2 cursor-not-allowed"
                >
                  <Bot size={14} /> Proceed to Results
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
