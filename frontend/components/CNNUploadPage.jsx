"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, ArrowRight, Lock, X, SlidersHorizontal, Image as ImageIcon } from "lucide-react";
import Navbar from "@/components/Navbar";

const DEFAULT_EPOCHS = 5;
const DEFAULT_BATCH_SIZE = 32;
const BATCH_OPTIONS = [8, 16, 32, 64, 128];

export default function CNNUploadPage() {
  const router = useRouter();
  const fileInputRef = useRef(null);

  const [dragActive, setDragActive] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH_SIZE);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file) => {
    setUploadedFile(file);
    setError("");
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const clearSelection = () => {
    setUploadedFile(null);
    setError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const validateInputs = () => {
    if (!uploadedFile) {
      return "Please upload a dataset zip file";
    }

    if (!uploadedFile.name.toLowerCase().endsWith(".zip")) {
      return "Dataset must be a .zip file";
    }

    if (Number.isNaN(Number(epochs)) || Number(epochs) < 1 || Number(epochs) > 50) {
      return "Epochs must be between 1 and 50";
    }

    if (!BATCH_OPTIONS.includes(Number(batchSize))) {
      return "Batch size must be one of: 8, 16, 32, 64, 128";
    }

    return null;
  };

  const handleStartTraining = async () => {
    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSubmitting(true);
    setError("");

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    try {
      const uploadFormData = new FormData();
      uploadFormData.append("file", uploadedFile);

      const uploadResponse = await fetch(`${apiUrl}/train/upload-dataset`, {
        method: "POST",
        body: uploadFormData,
      });

      if (!uploadResponse.ok) {
        const uploadError = await uploadResponse.json().catch(() => ({}));
        throw new Error(uploadError.detail || "Failed to upload dataset");
      }

      const uploadData = await uploadResponse.json();

      const startUrl = new URL(`${apiUrl}/train/start`);
      startUrl.searchParams.set("dataset_id", uploadData.dataset_id);
      startUrl.searchParams.set("epochs", String(epochs));
      startUrl.searchParams.set("batch_size", String(batchSize));

      const startResponse = await fetch(startUrl.toString(), {
        method: "POST",
      });

      if (!startResponse.ok) {
        const startError = await startResponse.json().catch(() => ({}));
        throw new Error(startError.detail || "Failed to start CNN training");
      }

      const startData = await startResponse.json();

      const trainingContext = {
        dataset_id: uploadData.dataset_id,
        class_count: uploadData.class_count,
        image_count: uploadData.image_count,
        job_id: startData.job_id,
        status: startData.status,
        message: startData.message,
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        uploaded_file_name: uploadedFile.name,
        started_at: Date.now(),
      };

      localStorage.setItem("cnnTrainingContext", JSON.stringify(trainingContext));
      router.push("/cnn/processing");
    } catch (err) {
      setError(err.message || "Failed to start training");
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Navbar className="z-50" />

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

        <div className="relative z-10 flex flex-col min-h-screen">
          <main className="flex-1 flex flex-col items-center justify-center p-6 lg:p-12">
            <div className="w-full max-w-5xl">
              <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-6">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest"
                      style={{
                        background: "rgba(117, 117, 117, 0.2)",
                        color: "#757575",
                      }}
                    >
                      CNN Pipeline
                    </span>
                    <Lock className="text-[#8C8C8C]" size={14} />
                  </div>
                  <h2 className="text-4xl md:text-5xl font-bold tracking-tighter text-white mb-2">
                    Upload Image Dataset
                  </h2>
                  <p className="text-[#8C8C8C] text-lg max-w-xl">
                    Upload a dataset zip, set default training parameters, and launch CNN model search.
                  </p>
                </div>
              </div>

              <div
                className="rounded-xl overflow-hidden"
                style={{
                  background: "rgba(117, 117, 117, 0.12)",
                  backdropFilter: "blur(20px)",
                  border: "1px solid rgba(140, 140, 140, 0.3)",
                  boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.4)",
                }}
              >
                <div className="grid lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-white/10">
                  <div className="p-8 lg:p-12 relative">
                    {uploadedFile && (
                      <button
                        onClick={clearSelection}
                        className="absolute top-4 right-4 p-2 rounded-lg transition-all z-10"
                        style={{
                          background: "rgba(255, 255, 255, 0.05)",
                          border: "1px solid rgba(255, 255, 255, 0.1)",
                        }}
                      >
                        <X className="text-[#8C8C8C]" size={16} />
                      </button>
                    )}

                    <div className="flex items-center gap-3 mb-8">
                      <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center"
                        style={{
                          background: "rgba(117, 117, 117, 0.1)",
                          border: "1px solid rgba(117, 117, 117, 0.3)",
                        }}
                      >
                        <Upload className="text-[#757575]" size={20} />
                      </div>
                      <h3 className="text-xl font-semibold text-white">Dataset Upload</h3>
                    </div>

                    <div
                      className={`group relative flex flex-col items-center justify-center py-16 px-6 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                        dragActive ? "border-[#757575] bg-white/10" : "border-[#8C8C8C]/40 bg-white/5"
                      }`}
                      onDragEnter={handleDrag}
                      onDragLeave={handleDrag}
                      onDragOver={handleDrag}
                      onDrop={handleDrop}
                      onClick={handleBrowseClick}
                    >
                      <Upload className="text-[#757575] mb-6" size={72} style={{ opacity: 0.8 }} />
                      <p className="text-white font-medium text-lg mb-1">
                        {uploadedFile ? uploadedFile.name : "Drag and drop dataset.zip"}
                      </p>
                      <p className="text-[#8C8C8C] text-sm text-center">
                        Required format: .zip with class subfolders and image files.
                      </p>
                      <button
                        className="mt-8 px-6 py-2.5 rounded-lg font-bold text-sm tracking-wide transition-transform hover:scale-105"
                        style={{
                          background: "#757575",
                          color: "white",
                          boxShadow: "0 10px 25px rgba(117, 117, 117, 0.2)",
                        }}
                      >
                        Browse Dataset Zip
                      </button>
                      <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileInput} accept=".zip" />
                    </div>
                  </div>

                  <div className="p-8 lg:p-12 flex flex-col">
                    <div className="flex items-center gap-3 mb-8">
                      <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center"
                        style={{
                          background: "rgba(176, 176, 176, 0.1)",
                          border: "1px solid rgba(176, 176, 176, 0.3)",
                        }}
                      >
                        <SlidersHorizontal className="text-[#B0B0B0]" size={20} />
                      </div>
                      <h3 className="text-xl font-semibold text-white">Training Parameters</h3>
                    </div>

                    <div className="space-y-6">
                      <div>
                        <label className="block text-xs font-bold text-[#8C8C8C] uppercase tracking-widest mb-2">Epochs</label>
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={epochs}
                          onChange={(e) => setEpochs(e.target.value)}
                          className="w-full rounded-lg py-4 px-4 text-white transition-all"
                          style={{
                            background: "rgba(255, 255, 255, 0.05)",
                            border: "1px solid rgba(140, 140, 140, 0.3)",
                            outline: "none",
                          }}
                        />
                        <p className="text-[10px] text-[#8C8C8C] mt-2">Default 5, valid range 1 to 50.</p>
                      </div>

                      <div>
                        <label className="block text-xs font-bold text-[#8C8C8C] uppercase tracking-widest mb-2">Batch Size</label>
                        <select
                          value={batchSize}
                          onChange={(e) => setBatchSize(Number(e.target.value))}
                          className="w-full rounded-lg py-4 px-4 text-white transition-all"
                          style={{
                            background: "rgba(255, 255, 255, 0.05)",
                            border: "1px solid rgba(140, 140, 140, 0.3)",
                            outline: "none",
                          }}
                        >
                          {BATCH_OPTIONS.map((option) => (
                            <option key={option} value={option} style={{ background: "#111111", color: "#E0E0E0" }}>
                              {option}
                            </option>
                          ))}
                        </select>
                        <p className="text-[10px] text-[#8C8C8C] mt-2">Default 32, allowed: 8, 16, 32, 64, 128.</p>
                      </div>
                    </div>

                    <div
                      className="mt-10 p-4 rounded-lg flex items-start gap-3"
                      style={{
                        background: "rgba(176, 176, 176, 0.05)",
                        border: "1px solid rgba(176, 176, 176, 0.2)",
                      }}
                    >
                      <ImageIcon className="text-[#B0B0B0] mt-0.5" size={18} />
                      <div>
                        <p className="text-xs font-bold text-[#B0B0B0] uppercase tracking-tight">Dataset Requirements</p>
                        <p className="text-[11px] text-[#8C8C8C] mt-1 leading-tight">
                          At least 2 class folders and at least 20 images total in the uploaded zip.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 flex flex-wrap items-center justify-between gap-6 px-4">
                {error ? (
                  <div
                    className="flex items-center gap-2 px-4 py-2 rounded-lg"
                    style={{
                      background: "rgba(220, 38, 38, 0.1)",
                      border: "1px solid rgba(220, 38, 38, 0.3)",
                    }}
                  >
                    <span className="text-red-400 text-sm">{error}</span>
                  </div>
                ) : (
                  <div className="text-[#8C8C8C] text-sm">Ready to run CNN architecture search and training.</div>
                )}

                <button
                  className="group flex items-center gap-3 px-8 py-4 rounded-xl font-bold tracking-tight transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    background: isSubmitting ? "#525252" : "#757575",
                    color: "white",
                    boxShadow: "0 0 30px rgba(117, 117, 117, 0.3)",
                  }}
                  onClick={handleStartTraining}
                  disabled={isSubmitting || !uploadedFile}
                >
                  {isSubmitting ? "Starting CNN Training..." : "Start CNN Training"}
                  <ArrowRight size={20} className={isSubmitting ? "animate-pulse" : "group-hover:translate-x-1 transition-transform"} />
                </button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
