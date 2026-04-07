"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingScreen from "@/components/LoadingScreen";

const POLL_INTERVAL_MS = 60000;

function mapStatusToStep(message, progress) {
  const lowerMsg = (message || "").toLowerCase();

  if (lowerMsg.includes("preprocess") || progress < 20) {
    return { currentStep: "Dataset Validation", stepNumber: 1 };
  }
  if (lowerMsg.includes("train") || progress < 45) {
    return { currentStep: "CNN Training", stepNumber: 2 };
  }
  if (lowerMsg.includes("architecture") || progress < 70) {
    return { currentStep: "Architecture Selection", stepNumber: 3 };
  }
  if (lowerMsg.includes("validation") || progress < 90) {
    return { currentStep: "Validation", stepNumber: 4 };
  }
  return { currentStep: "Finalizing Best Model", stepNumber: 5 };
}

function formatElapsed(startedAt) {
  if (!startedAt) return "00:00:00";
  const elapsedMs = Math.max(0, Date.now() - startedAt);
  const totalSeconds = Math.floor(elapsedMs / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

export default function CNNProcessingPage() {
  const router = useRouter();
  const [context, setContext] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("Initializing CNN pipeline");
  const [statusValue, setStatusValue] = useState("running");
  const [currentStep, setCurrentStep] = useState("Dataset Validation");
  const [stepNumber, setStepNumber] = useState(1);
  const [elapsedTime, setElapsedTime] = useState("00:00:00");
  const [error, setError] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    const storedContext = localStorage.getItem("cnnTrainingContext");
    if (!storedContext) {
      router.replace("/cnn");
      return;
    }

    const parsedContext = JSON.parse(storedContext);
    setContext(parsedContext);
    setElapsedTime(formatElapsed(parsedContext.started_at));
  }, [router]);

  useEffect(() => {
    if (!context?.job_id) return;

    let isMounted = true;

    const pollStatus = async () => {
      try {
        const response = await fetch(`${apiUrl}/train/status/${context.job_id}`);
        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || "Failed to fetch training status");
        }

        const data = await response.json();
        if (!isMounted) return;

        const safeProgress = Math.min(100, Math.max(0, Number(data.progress || 0)));
        const message = data.message || "Processing CNN training";

        setProgress(safeProgress);
        setStatusMessage(message);
        setStatusValue(data.status || "running");
        setElapsedTime(formatElapsed(context.started_at));

        const mapped = mapStatusToStep(message, safeProgress);
        setCurrentStep(mapped.currentStep);
        setStepNumber(mapped.stepNumber);

        if (data.status === "failed") {
          setError(data.error || "CNN training failed");
          return;
        }

        if (data.status === "completed") {
          const resultResponse = await fetch(`${apiUrl}/train/result/${context.job_id}`);
          if (!resultResponse.ok) {
            const resultErr = await resultResponse.json().catch(() => ({}));
            throw new Error(resultErr.detail || "Failed to fetch final training result");
          }

          const resultData = await resultResponse.json();
          localStorage.setItem("cnnTrainingResult", JSON.stringify(resultData));
          router.replace("/cnn/result");
        }
      } catch (err) {
        if (!isMounted) return;
        setError(err.message || "Error while tracking training status");
      }
    };

    pollStatus();
    const intervalId = setInterval(pollStatus, POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, [apiUrl, context, router]);

  const estimatedTime = useMemo(() => {
    if (statusValue === "completed") return "Complete";
    if (statusValue === "failed") return "Stopped";
    return "Updated every 1 minute";
  }, [statusValue]);

  const modelName = context?.uploaded_file_name || "Image Dataset";

  return (
    <div>
      {error ? (
        <div className="fixed inset-x-0 top-20 z-[60] flex justify-center px-6">
          <div
            className="max-w-2xl w-full rounded-lg px-4 py-3 text-sm text-red-300"
            style={{
              background: "rgba(127, 29, 29, 0.6)",
              border: "1px solid rgba(248, 113, 113, 0.5)",
            }}
          >
            {error}
          </div>
        </div>
      ) : null}

      <LoadingScreen
        progress={parseFloat(progress.toFixed(1))}
        modelName={modelName}
        currentStep={currentStep}
        stepNumber={stepNumber}
        totalSteps={5}
        accuracy={Math.min(60 + progress * 0.35, 98)}
        validationLoss={parseFloat((0.6 - progress * 0.005).toFixed(3))}
        gpuUtilization={Math.min(68 + progress * 0.22, 97)}
        memoryBandwidth={Math.min(60 + progress * 0.3, 98)}
        elapsedTime={elapsedTime}
        estimatedTime={estimatedTime}
        features={String(context?.image_count || "N/A")}
        isCompleted={false}
      />

      <div className="fixed bottom-4 right-6 z-[60] px-4 py-2 rounded-lg text-xs"
        style={{
          background: "rgba(0, 0, 0, 0.5)",
          border: "1px solid rgba(140, 140, 140, 0.3)",
          color: "#B0B0B0",
        }}
      >
        Status: {statusValue} | {statusMessage}
      </div>
    </div>
  );
}
