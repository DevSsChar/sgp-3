"use client";
import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Upload, Database, CloudUpload, Link as LinkIcon, Download, ArrowRight, Lock, Shield, X, Target } from 'lucide-react';
import Navbar from '@/components/Navbar';
import LoadingScreen from '@/components/LoadingScreen';

export default function UploadPage() {
  const router = useRouter();
  const [kaggleUrl, setKaggleUrl] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState(null);
  const [activeMethod, setActiveMethod] = useState(null); // 'local' or 'kaggle'
  const [targetColumn, setTargetColumn] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [trainingData, setTrainingData] = useState(null);
  const [elapsedTime, setElapsedTime] = useState("00:00:00");
  const [currentStep, setCurrentStep] = useState("Data Preprocessing");
  const [stepNumber, setStepNumber] = useState(1);
  const [trainingComplete, setTrainingComplete] = useState(false); // Training timer complete
  const fileInputRef = useRef(null);

  // Training progress simulation
  useEffect(() => {
    if (!isTraining) return;

    const startTime = Date.now();
    const totalDuration = 15 * 60 * 1000; // 15 minutes to reach 100%

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      let progressPercent = Math.min((elapsed / totalDuration) * 100, 100);

      setProgress(progressPercent);

      // Update elapsed time
      const seconds = Math.floor(elapsed / 1000);
      const mins = Math.floor(seconds / 60);
      const hrs = Math.floor(mins / 60);
      setElapsedTime(
        `${String(hrs).padStart(2, '0')}:${String(mins % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
      );

      // Update steps based on progress
      if (progressPercent < 20) {
        setCurrentStep("Data Preprocessing");
        setStepNumber(1);
      } else if (progressPercent < 40) {
        setCurrentStep("Feature Engineering");
        setStepNumber(2);
      } else if (progressPercent < 60) {
        setCurrentStep("Model Training");
        setStepNumber(3);
      } else if (progressPercent < 80) {
        setCurrentStep("Hyperparameter Tuning");
        setStepNumber(4);
      } else if (progressPercent < 95) {
        setCurrentStep("Model Validation");
        setStepNumber(5);
      } else {
        setCurrentStep("Finalizing Best Model");
        setStepNumber(6);
      }

      // When progress reaches 100%, mark as complete
      if (progressPercent >= 100) {
        setTrainingComplete(true);
        clearInterval(progressInterval);
      }
    }, 1000);

    return () => clearInterval(progressInterval);
  }, [isTraining]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (activeMethod === 'kaggle') return; // Prevent drag if Kaggle is active
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
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = (files) => {
    console.log('Files uploaded:', files);
    setUploadedFiles(files);
    setActiveMethod('local');
    setKaggleUrl(''); // Clear Kaggle URL when file is uploaded
  };

  const handleBrowseClick = () => {
    if (activeMethod === 'kaggle') return; // Prevent browse if Kaggle is active
    fileInputRef.current?.click();
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const handleKaggleFetch = () => {
    console.log('Fetching from Kaggle:', kaggleUrl);
    // Handle Kaggle fetch logic here
  };

  const handleKaggleUrlChange = (value) => {
    setKaggleUrl(value);
    if (value.trim()) {
      setActiveMethod('kaggle');
      setUploadedFiles(null); // Clear uploaded files when Kaggle URL is entered
    } else {
      setActiveMethod(null);
    }
  };

  const handleStartIngestion = async () => {
    if (!targetColumn.trim()) {
      setError('Please specify a target column name');
      return;
    }

    if (!activeMethod) {
      setError('Please select a data source (upload file or enter Kaggle URL)');
      return;
    }

    setIsSubmitting(true);
    setError('');

    // Store initial training info before API call
    const initialTrainingInfo = {
      startTime: Date.now(),
      targetColumn: targetColumn,
      method: activeMethod,
      fileName: uploadedFiles ? uploadedFiles[0]?.name : null,
      kaggleUrl: activeMethod === 'kaggle' ? kaggleUrl : null
    };
    setTrainingData(initialTrainingInfo);
    localStorage.setItem('trainingData', JSON.stringify(initialTrainingInfo));

    // Start training mode immediately to show LoadingScreen
    setIsTraining(true);

    try {
      let response;
      
      if (activeMethod === 'local') {
        // File upload endpoint
        if (!uploadedFiles || uploadedFiles.length === 0) {
          setError('Please upload a file');
          setIsSubmitting(false);
          setIsTraining(false);
          return;
        }

        const formData = new FormData();
        formData.append('file', uploadedFiles[0]);
        formData.append('target_column', targetColumn);

        response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/train`, {
          method: 'POST',
          body: formData,
        });
      } else if (activeMethod === 'kaggle') {
        // Kaggle URL endpoint
        if (!kaggleUrl.trim()) {
          setError('Please enter a Kaggle dataset URL');
          setIsSubmitting(false);
          setIsTraining(false);
          return;
        }

        response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/train/kaggle`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            kaggle_url: kaggleUrl,
            target_column: targetColumn,
          }),
        });
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Training failed');
      }

      const data = await response.json();
      
      // Update training info with API response data
      const updatedTrainingInfo = {
        ...trainingData,
        ...data
      };
      setTrainingData(updatedTrainingInfo);
      localStorage.setItem('trainingData', JSON.stringify(updatedTrainingInfo));

      // Training complete - show "See Model" button immediately
      setTrainingComplete(true);
      setProgress(100);
      setIsSubmitting(false);
    } catch (err) {
      console.error('Training error:', err);
      setError(err.message || 'Failed to start training. Please check your backend is running.');
      setIsSubmitting(false);
      setIsTraining(false); // Revert to upload page on error
    }
  };

  const clearSelection = () => {
    setActiveMethod(null);
    setUploadedFiles(null);
    setKaggleUrl('');
    setTargetColumn('');
    setError('');
  };

  const handlePause = () => {
    console.log("Training paused");
  };

  const handleViewLogs = () => {
    console.log("Opening logs...");
  };

  const handleSeeModel = async () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    try {
      // Fetch best model when user clicks button
      const bestModelResponse = await fetch(`${apiUrl}/models/best`);
      if (!bestModelResponse.ok) {
        throw new Error('Model not ready yet. Please wait a moment and try again.');
      }
      
      const bestModelData = await bestModelResponse.json();
      localStorage.setItem('modelResults', JSON.stringify(bestModelData));

      // Fetch all models
      try {
        const allModelsResponse = await fetch(`${apiUrl}/models`);
        if (allModelsResponse.ok) {
          const allModelsData = await allModelsResponse.json();
          localStorage.setItem('allModels', JSON.stringify(allModelsData));
        }
      } catch (err) {
        console.log('Could not fetch all models, will use best model only');
      }
      
      // Navigate to results page
      router.push('/results');
    } catch (error) {
      console.error('Error fetching model data:', error);
      alert(error.message || 'Failed to fetch model results. Please try again in a moment.');
    }
  };

  const estimatedRemaining = () => {
    if (isCompleted) return "Complete";
    const remaining = Math.max(0, 15 - Math.floor((progress / 100) * 15));
    return `~${String(Math.floor(remaining / 60)).padStart(2, '0')}:${String(remaining % 60).padStart(2, '0')}:00`;
  };

  // If training is in progress, show LoadingScreen
  if (isTraining) {
    return (
      <LoadingScreen
        progress={parseFloat(progress.toFixed(1))}
        modelName={trainingData?.method === 'kaggle' ? 'Kaggle Dataset' : uploadedFiles ? uploadedFiles[0]?.name : 'Uploaded CSV'}
        currentStep={currentStep}
        stepNumber={stepNumber}
        totalSteps={6}
        accuracy={Math.min(85 + progress * 0.1, 95.8)}
        validationLoss={parseFloat((0.2 - (progress * 0.0015)).toFixed(3))}
        gpuUtilization={Math.min(75 + progress * 0.15, 92)}
        memoryBandwidth={Math.min(80 + progress * 0.12, 95)}
        elapsedTime={elapsedTime}
        estimatedTime={estimatedRemaining()}
        features="1,248"
        onPause={handlePause}
        onViewLogs={handleViewLogs}
        isCompleted={trainingComplete}
        onSeeModel={handleSeeModel}
      />
    );
  }

  return (
    <>
    <Navbar className="z-50" />

    <div className="min-h-screen relative overflow-hidden" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
      {/* Background Layer */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-[#2D2D2D] via-[#000000] to-[#2D2D2D]">
          <div 
            className="absolute inset-0"
            style={{
              backgroundImage: 'linear-gradient(to right, rgba(140, 140, 140, 0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(140, 140, 140, 0.05) 1px, transparent 1px)',
              backgroundSize: '40px 40px'
            }}
          />
          
          {/* Decorative Particles */}
          {[
            { top: '25%', left: '25%' },
            { top: '33%', right: '25%' },
            { bottom: '25%', left: '33%' },
            { bottom: '50%', right: '10%' }
          ].map((pos, i) => (
            <div 
              key={i}
              className="absolute w-0.5 h-0.5 rounded-full"
              style={{ 
                background: 'rgba(117, 117, 117, 0.4)',
                ...pos,
                filter: 'drop-shadow(0 0 2px rgba(117, 117, 117, 0.4))'
              }}
            />
          ))}
        </div>
        
        {/* Background Glow Overlays */}
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] rounded-full blur-[120px]" style={{ background: 'rgba(117, 117, 117, 0.1)' }} />
        <div className="absolute -bottom-[10%] -right-[10%] w-[30%] h-[30%] rounded-full blur-[100px]" style={{ background: 'rgba(140, 140, 140, 0.05)' }} />
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex flex-col min-h-screen">

        {/* Main Workspace */}
        <main className="flex-1 flex flex-col items-center justify-center p-6 lg:p-12">
          <div className="w-full max-w-5xl">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-6">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span 
                    className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-widest"
                    style={{
                      background: 'rgba(117, 117, 117, 0.2)',
                      color: '#757575'
                    }}
                  >
                    Enterprise Tier
                  </span>
                  <Lock className="text-[#8C8C8C]" size={14} />
                </div>
                <h2 className="text-4xl md:text-5xl font-bold tracking-tighter text-white mb-2">
                  Ingest Training Data
                </h2>
                <p className="text-[#8C8C8C] text-lg max-w-xl">
                  Select a data source to begin the automated feature engineering pipeline.
                </p>
              </div>
              <div className="flex gap-4">
                {/* <button 
                  className="flex items-center gap-2 px-6 py-3 rounded-xl transition-all text-white font-medium"
                  style={{
                    background: 'rgba(255, 255, 255, 0.05)',
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)'}
                >
                  <CloudUpload size={16} />
                  Cloud Storage
                </button> */}
              </div>
            </div>

            {/* Central Glassmorphism Panel */}
            <div 
              className="rounded-xl overflow-hidden"
              style={{
                background: 'rgba(117, 117, 117, 0.12)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(140, 140, 140, 0.3)',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.4)'
              }}
            >
              <div className="grid lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-white/10">
                {/* Section 1: Local Upload */}
                <div className={`p-8 lg:p-12 transition-opacity relative ${activeMethod === 'kaggle' ? 'opacity-40 pointer-events-none' : ''}`}>
                  {activeMethod === 'local' && (
                    <button
                      onClick={clearSelection}
                      className="absolute top-4 right-4 p-2 rounded-lg transition-all z-10"
                      style={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.1)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 0, 0, 0.1)';
                        e.currentTarget.style.borderColor = 'rgba(255, 0, 0, 0.3)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                      }}
                    >
                      <X className="text-[#8C8C8C]" size={16} />
                    </button>
                  )}
                  <div className="flex items-center gap-3 mb-8">
                    <div 
                      className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{
                        background: 'rgba(117, 117, 117, 0.1)',
                        border: '1px solid rgba(117, 117, 117, 0.3)'
                      }}
                    >
                      <Upload className="text-[#757575]" size={20} />
                    </div>
                    <h3 className="text-xl font-semibold text-white">Upload Local File</h3>
                  </div>

                  <div 
                    className={`group relative flex flex-col items-center justify-center py-16 px-6 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                      dragActive ? 'border-[#757575] bg-white/10' : 'border-[#8C8C8C]/40 bg-white/5'
                    }`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={handleBrowseClick}
                    style={{
                      transition: 'all 0.3s'
                    }}
                    onMouseEnter={(e) => {
                      if (!dragActive) {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';
                        e.currentTarget.style.borderColor = 'rgba(117, 117, 117, 0.5)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!dragActive) {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.borderColor = 'rgba(140, 140, 140, 0.4)';
                      }
                    }}
                  >
                    <div 
                      className="mb-6"
                      style={{
                        filter: 'drop-shadow(0 0 15px rgba(117, 117, 117, 0.4))'
                      }}
                    >
                      <Upload 
                        className="text-[#757575] group-hover:opacity-100 transition-opacity" 
                        size={72}
                        style={{ opacity: 0.8 }}
                      />
                    </div>
                    <p className="text-white font-medium text-lg mb-1">
                      {uploadedFiles ? `${uploadedFiles.length} file(s) selected` : 'Drag & drop local files here'}
                    </p>
                    <p className="text-[#8C8C8C] text-sm text-center">
                      {uploadedFiles ? Array.from(uploadedFiles).map(f => f.name).join(', ') : 'Supported formats: CSV, Parquet, JSON, Zip (Max 10GB)'}
                    </p>
                    <button 
                      className="mt-8 px-6 py-2.5 rounded-lg font-bold text-sm tracking-wide transition-transform hover:scale-105"
                      style={{
                        background: '#757575',
                        color: 'white',
                        boxShadow: '0 10px 25px rgba(117, 117, 117, 0.2)'
                      }}
                    >
                      Browse Files
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      className="hidden"
                      onChange={handleFileInput}
                      accept=".csv,.parquet,.json,.zip"
                    />
                  </div>
                </div>

                {/* Section 2: Kaggle Dataset */}
                <div className={`p-8 lg:p-12 flex flex-col transition-opacity relative ${activeMethod === 'local' ? 'opacity-40 pointer-events-none' : ''}`}>
                  {activeMethod === 'kaggle' && (
                    <button
                      onClick={clearSelection}
                      className="absolute top-4 right-4 p-2 rounded-lg transition-all z-10"
                      style={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.1)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 0, 0, 0.1)';
                        e.currentTarget.style.borderColor = 'rgba(255, 0, 0, 0.3)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                      }}
                    >
                      <X className="text-[#8C8C8C]" size={16} />
                    </button>
                  )}
                  <div className="flex items-center gap-3 mb-8">
                    <div 
                      className="w-10 h-10 rounded-lg flex items-center justify-center"
                      style={{
                        background: 'rgba(176, 176, 176, 0.1)',
                        border: '1px solid rgba(176, 176, 176, 0.3)'
                      }}
                    >
                      <Database className="text-[#B0B0B0]" size={20} />
                    </div>
                    <h3 className="text-xl font-semibold text-white">Kaggle Dataset URL</h3>
                  </div>

                  <div className="flex-1 flex flex-col justify-between">
                    <div className="space-y-6">
                      <p className="text-[#8C8C8C] text-sm leading-relaxed">
                        Securely fetch public or private datasets directly from Kaggle. Ensure your Kaggle API key is configured in your profile settings.
                      </p>

                      <div className="space-y-4">
                        <div>
                          <label className="block text-xs font-bold text-[#8C8C8C] uppercase tracking-widest mb-2">
                            Dataset Reference
                          </label>
                          <div className="relative group">
                            <input
                              type="text"
                              className="w-full rounded-lg py-4 pl-4 pr-12 text-white transition-all"
                              style={{
                                background: 'rgba(255, 255, 255, 0.05)',
                                border: '1px solid rgba(140, 140, 140, 0.3)',
                                outline: 'none'
                              }}
                              placeholder="username/dataset-name"
                              value={kaggleUrl}
                              onChange={(e) => handleKaggleUrlChange(e.target.value)}
                              onFocus={(e) => e.target.style.borderColor = '#757575'}
                              onBlur={(e) => e.target.style.borderColor = 'rgba(140, 140, 140, 0.3)'}
                            />
                            <LinkIcon 
                              className="absolute right-4 top-1/2 -translate-y-1/2 text-[#8C8C8C] group-focus-within:text-[#757575] transition-colors"
                              size={20}
                            />
                          </div>
                        </div>

                        {/* <button 
                          className="w-full py-4 rounded-lg text-white font-bold transition-all flex items-center justify-center gap-2"
                          style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            boxShadow: 'inset 0 0 10px rgba(255, 255, 255, 0.1)'
                          }}
                          onClick={handleKaggleFetch}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#757575';
                            e.currentTarget.style.borderColor = '#757575';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                            e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                          }}
                        >
                          <Download size={16} />
                          Fetch Dataset
                        </button> */}
                      </div>
                    </div>

                    <div 
                      className="mt-12 p-4 rounded-lg flex items-start gap-3"
                      style={{
                        background: 'rgba(176, 176, 176, 0.05)',
                        border: '1px solid rgba(176, 176, 176, 0.2)'
                      }}
                    >
                      {/* <Shield className="text-[#B0B0B0]" size={20} /> */}
                      <div>
                        <p className="text-xs font-bold text-[#B0B0B0] uppercase tracking-tight">
                          Refrence of URL :
                        </p>
                        <p className="text-[10px] text-[#8C8C8C] mt-1 leading-tight">
                          https://www.kaggle.com/datasets/nikhil1e9/loan-default
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Target Column Field - Shows when method is selected */}
            {activeMethod && (
              <div 
                className="mt-6 rounded-xl p-6 transition-all duration-300"
                style={{
                  background: 'rgba(117, 117, 117, 0.1)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(140, 140, 140, 0.3)',
                  boxShadow: '0 10px 30px -10px rgba(0, 0, 0, 0.3)'
                }}
              >
                <div className="flex items-start gap-4">
                  <div 
                    className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{
                      background: 'rgba(117, 117, 117, 0.2)',
                      border: '1px solid rgba(117, 117, 117, 0.3)'
                    }}
                  >
                    <Target className="text-[#757575]" size={20} />
                  </div>
                  <div className="flex-1">
                    <label className="block text-sm font-bold text-white mb-2">
                      Target Column Name
                    </label>
                    <p className="text-xs text-[#8C8C8C] mb-4">
                      Specify the column name that contains the target variable for model training.
                    </p>
                    <div className="relative">
                      <input
                        type="text"
                        className="w-full rounded-lg py-3 px-4 text-white transition-all"
                        style={{
                          background: 'rgba(255, 255, 255, 0.05)',
                          border: '1px solid rgba(140, 140, 140, 0.3)',
                          outline: 'none'
                        }}
                        placeholder="e.g., price, class, label, target"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        onFocus={(e) => e.target.style.borderColor = '#757575'}
                        onBlur={(e) => e.target.style.borderColor = 'rgba(140, 140, 140, 0.3)'}  
                      />
                      {targetColumn && (
                        <div 
                          className="absolute right-3 top-1/2 -translate-y-1/2 px-2 py-0.5 rounded text-[10px] font-bold"
                          style={{
                            background: 'rgba(117, 117, 117, 0.3)',
                            color: '#B0B0B0'
                          }}
                        >
                          ✓ Set
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Footer Stats */}
            <div className="mt-8 flex flex-wrap items-center justify-between gap-6 px-4">
              <div className="flex items-center gap-8">
                {error && (
                  <div 
                    className="flex items-center gap-2 px-4 py-2 rounded-lg"
                    style={{
                      background: 'rgba(220, 38, 38, 0.1)',
                      border: '1px solid rgba(220, 38, 38, 0.3)'
                    }}
                  >
                    <span className="text-red-400 text-sm">{error}</span>
                  </div>
                )}
                {!error && (
                  <>
                    <div>
                      <p className="text-[10px] font-bold text-[#8C8C8C] uppercase tracking-widest mb-1">
                        Avg. Ingestion Speed
                      </p>
                      <p className="text-white font-display font-medium">1.2 GB/s</p>
                    </div>
                    <div className="h-8 w-px bg-white/10"></div>
                    <div>
                      <p className="text-[10px] font-bold text-[#8C8C8C] uppercase tracking-widest mb-1">
                        Queue Status
                      </p>
                      <p className="text-[#B0B0B0] font-display font-medium flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-[#B0B0B0]"></span>
                        Nominal
                      </p>
                    </div>
                  </>
                )}
              </div>

              <button 
                className="group flex items-center gap-3 px-8 py-4 rounded-xl font-bold tracking-tight transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  background: isSubmitting ? '#525252' : '#757575',
                  color: 'white',
                  boxShadow: '0 0 30px rgba(117, 117, 117, 0.3)'
                }}
                onClick={handleStartIngestion}
                disabled={isSubmitting || !activeMethod || !targetColumn.trim()}
                onMouseEnter={(e) => {
                  if (!isSubmitting && activeMethod && targetColumn.trim()) {
                    e.currentTarget.style.boxShadow = '0 0 30px rgba(117, 117, 117, 0.5)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = '0 0 30px rgba(117, 117, 117, 0.3)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                {isSubmitting ? 'Starting Training...' : 'Start Ingestion Process'}
                <ArrowRight className={`transition-transform ${isSubmitting ? 'animate-pulse' : 'group-hover:translate-x-1'}`} size={20} />
              </button>
            </div>
          </div>
        </main>

        {/* Sidebar Floating Elements */}
        <div className="fixed left-6 top-1/2 -translate-y-1/2 hidden xl:flex flex-col gap-6">
          <div className="flex flex-col items-center gap-2">
            <span 
              className="text-[10px] font-bold uppercase tracking-[0.3em]"
              style={{ 
                writingMode: 'vertical-rl',
                transform: 'rotate(180deg)',
                color: '#757575'
              }}
            >
              Phase 01
            </span>
            <div 
              className="w-1 h-12 rounded-full"
              style={{
                background: 'linear-gradient(to bottom, #757575, transparent)'
              }}
            ></div>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
