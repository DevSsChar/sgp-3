"use client";
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from '@/components/Navbar';
import Sidebar from '@/components/Sidebar';
import AllModelsView from '@/components/AllModelsView';
import ModelDetailView from '@/components/ModelDetailView';

// Mock data for all models
const MOCK_ALL_MODELS = [
  {
    model_id: "GradientBoosting_b22de4e9f16e_1771603320",
    model_name: "GradientBoosting",
    task: "classification",
    cv_score_mean: 0.9963594852635949,
    cv_score_std: 0.0034026260909819847,
    test_score: 1,
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
    timestamp: "2026-02-20T21:32:00.990119",
    feature_count: 136,
    train_samples: 1097,
    pickle_download: "/models/GradientBoosting_b22de4e9f16e_1771603320/download"
  },
  {
    model_id: "GradientBoosting_b22de4e9f16e_1771615752",
    model_name: "GradientBoosting",
    task: "classification",
    cv_score_mean: 0.9972685761726858,
    cv_score_std: 0.0022302110935331866,
    test_score: 0.9963636363636363,
    latency_ms: 0.0053310394287109375,
    size_mb: 3,
    final_score: 0.8463631032596934,
    hyperparameters: {
      n_estimators: 195,
      learning_rate: 0.07300813208006586,
      max_depth: 5,
      min_samples_split: 5,
      subsample: 0.7288875696068378
    },
    dataset_hash: "b22de4e9f16e",
    timestamp: "2026-02-21T00:59:12.944102",
    feature_count: 136,
    train_samples: 1097,
    pickle_download: "/models/GradientBoosting_b22de4e9f16e_1771615752/download"
  }
];

export default function ResultsPage() {
  const router = useRouter();
  const [modelData, setModelData] = useState(null);
  const [allModels, setAllModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('best');
  const [selectedModel, setSelectedModel] = useState(null);
  const [dataFetched, setDataFetched] = useState({
    bestModel: false,
    allModels: false
  });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Fetch best model from API
  const fetchBestModel = async () => {
    try {
      const response = await fetch(`${apiUrl}/models/best`);
      if (response.ok) {
        const data = await response.json();
        setModelData(data);
        localStorage.setItem('modelResults', JSON.stringify(data));
        setDataFetched(prev => ({ ...prev, bestModel: true }));
        return data;
      }
    } catch (error) {
      console.error('Error fetching best model:', error);
    }
    return null;
  };

  // Fetch all models from API
  const fetchAllModels = async () => {
    try {
      const response = await fetch(`${apiUrl}/models`);
      if (response.ok) {
        const data = await response.json();
        setAllModels(data);
        localStorage.setItem('allModels', JSON.stringify(data));
        setDataFetched(prev => ({ ...prev, allModels: true }));
        return data;
      }
    } catch (error) {
      console.error('Error fetching all models:', error);
    }
    return null;
  };

  useEffect(() => {
    const loadData = async () => {
      // First, try to load from localStorage
      const storedData = localStorage.getItem('modelResults');
      const storedAllModels = localStorage.getItem('allModels');
      
      let hasStoredBest = false;
      let hasStoredAll = false;

      if (storedData) {
        try {
          const parsed = JSON.parse(storedData);
          setModelData(parsed);
          setDataFetched(prev => ({ ...prev, bestModel: true }));
          hasStoredBest = true;
        } catch (error) {
          console.error('Error parsing model data:', error);
        }
      }

      if (storedAllModels) {
        try {
          const parsed = JSON.parse(storedAllModels);
          setAllModels(parsed);
          setDataFetched(prev => ({ ...prev, allModels: true }));
          hasStoredAll = true;
        } catch (error) {
          console.error('Error parsing all models:', error);
        }
      }

      // If no stored data, fetch from API
      if (!hasStoredBest) {
        const bestModel = await fetchBestModel();
        if (!bestModel) {
          // Fallback to mock data if API fails
          setModelData(MOCK_ALL_MODELS[0]);
        }
      }

      if (!hasStoredAll) {
        const models = await fetchAllModels();
        if (!models || models.length === 0) {
          // Fallback to mock data if API fails
          setAllModels(MOCK_ALL_MODELS);
        }
      }
      
      setLoading(false);
    };

    loadData();
  }, []);

  const handleSelectModel = (model) => {
    setSelectedModel(model);
    setActiveView('best');
  };

  const handleViewChange = (view) => {
    setActiveView(view);
    if (view === 'best') {
      setSelectedModel(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#000000] flex items-center justify-center">
        <div className="text-[#B0B0B0] text-lg">Loading results...</div>
      </div>
    );
  }

  const bestModel = modelData || MOCK_ALL_MODELS[0];
  const displayModel = selectedModel || bestModel;

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

      {/* Main Layout with Sidebar */}
      <div className="relative z-10 flex">
        <Sidebar activeView={activeView} onViewChange={handleViewChange} />

        {/* Main Content Area */}
        <main className="flex-1 p-8">
          {activeView === 'all' ? (
            <AllModelsView modelsData={allModels} onSelectModel={handleSelectModel} />
          ) : (
            <ModelDetailView modelData={displayModel} />
          )}
        </main>
      </div>
    </div>
  );
}
