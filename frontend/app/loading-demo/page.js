"use client";
import LoadingScreen from "@/components/LoadingScreen";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LoadingDemo() {
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);
  const [trainingData, setTrainingData] = useState(null);
  const [elapsedTime, setElapsedTime] = useState("00:00:00");
  const [currentStep, setCurrentStep] = useState("Data Preprocessing");
  const [stepNumber, setStepNumber] = useState(1);

  useEffect(() => {
    // Load training data from localStorage
    const storedData = localStorage.getItem('trainingData');
    if (storedData) {
      setTrainingData(JSON.parse(storedData));
    }

    // Simulate training progress over 15 minutes
    const startTime = Date.now();
    const totalDuration = 15 * 60 * 1000; // 15 minutes in milliseconds

    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progressPercent = Math.min((elapsed / totalDuration) * 100, 100);
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

      // Mark as completed after 15 minutes
      if (elapsed >= totalDuration) {
        setIsCompleted(true);
        clearInterval(progressInterval);
      }
    }, 1000);

    return () => clearInterval(progressInterval);
  }, []);

  // Fetch model data when training completes
  useEffect(() => {
    if (isCompleted) {
      const fetchModelData = async () => {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        
        try {
          // Fetch best model
          const bestModelResponse = await fetch(`${apiUrl}/models/best`);
          if (bestModelResponse.ok) {
            const bestModelData = await bestModelResponse.json();
            localStorage.setItem('modelResults', JSON.stringify(bestModelData));
          }

          // Fetch all models
          const allModelsResponse = await fetch(`${apiUrl}/models`);
          if (allModelsResponse.ok) {
            const allModelsData = await allModelsResponse.json();
            localStorage.setItem('allModels', JSON.stringify(allModelsData));
          }
        } catch (error) {
          console.error('Error fetching model data on completion:', error);
        }
      };

      fetchModelData();
    }
  }, [isCompleted]);

  const handlePause = () => {
    console.log("Training paused");
  };

  const handleViewLogs = () => {
    console.log("Opening logs...");
  };

  const handleSeeModel = async () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    try {
      // Fetch best model
      const bestModelResponse = await fetch(`${apiUrl}/models/best`);
      if (bestModelResponse.ok) {
        const bestModelData = await bestModelResponse.json();
        localStorage.setItem('modelResults', JSON.stringify(bestModelData));
      }

      // Fetch all models
      const allModelsResponse = await fetch(`${apiUrl}/models`);
      if (allModelsResponse.ok) {
        const allModelsData = await allModelsResponse.json();
        localStorage.setItem('allModels', JSON.stringify(allModelsData));
      }
    } catch (error) {
      console.error('Error fetching model data:', error);
      
      // Fallback to mock data if API is unavailable
      const mockModelData = {
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
      localStorage.setItem('modelResults', JSON.stringify(mockModelData));
    }
    
    // Navigate to results page
    router.push('/results');
  };

  // Calculate estimated time remaining
  const estimatedRemaining = () => {
    if (isCompleted) return "Complete";
    const remaining = Math.max(0, 15 - Math.floor((progress / 100) * 15));
    return `~${String(Math.floor(remaining / 60)).padStart(2, '0')}:${String(remaining % 60).padStart(2, '0')}:00`;
  };

  return (
    <LoadingScreen
      progress={parseFloat(progress.toFixed(1))}
      modelName={trainingData?.method === 'kaggle' ? 'Kaggle Dataset' : 'Uploaded CSV'}
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
      isCompleted={isCompleted}
      onSeeModel={handleSeeModel}
    />
  );
}
