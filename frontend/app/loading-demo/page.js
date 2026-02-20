"use client";
import LoadingScreen from "@/components/LoadingScreen";
import { useState } from "react";

export default function LoadingDemo() {
  const [progress, setProgress] = useState(68.4);

  // Simulate progress increase
  const handlePause = () => {
    console.log("Training paused");
  };

  const handleViewLogs = () => {
    console.log("Opening logs...");
  };

  return (
    <LoadingScreen
      progress={progress}
      modelName=""
      currentStep="Hyperparameter Tuning"
      stepNumber={4}
      totalSteps={6}
      accuracy={94.2}
      validationLoss={0.128}
      gpuUtilization={88}
      memoryBandwidth={92}
      elapsedTime="00:42:15"
      estimatedTime="~00:15:00"
      features="1,248"
      onPause={handlePause}
      onViewLogs={handleViewLogs}
    />
  );
}
