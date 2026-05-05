"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import CNNResultsView from "@/components/CNNResultsView";

export default function CNNResultPage() {
  const router = useRouter();
  const [resultData, setResultData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const storedResult = localStorage.getItem("cnnTrainingResult");
    if (storedResult) {
      try {
        setResultData(JSON.parse(storedResult));
        return;
      } catch {
        localStorage.removeItem("cnnTrainingResult");
      }
    }

    const context = localStorage.getItem("cnnTrainingContext");
    if (!context) {
      router.replace("/cnn");
      return;
    }

    const parsedContext = JSON.parse(context);
    if (!parsedContext.job_id) {
      router.replace("/cnn");
      return;
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

    const loadResult = async () => {
      try {
        const response = await fetch(`${apiUrl}/train/result/${parsedContext.job_id}`);
        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || "Could not fetch CNN result");
        }

        const data = await response.json();
        if (data.status !== "completed") {
          router.replace("/cnn/processing");
          return;
        }

        localStorage.setItem("cnnTrainingResult", JSON.stringify(data));
        setResultData(data);
      } catch (err) {
        setError(err.message || "Failed to load result");
      }
    };

    loadResult();
  }, [router]);

  if (error) {
    return (
      <div className="min-h-screen bg-[#000000] flex items-center justify-center p-6 text-center">
        <div>
          <p className="text-red-300 text-lg">{error}</p>
          <button
            onClick={() => router.push("/cnn")}
            className="mt-6 px-5 py-2 rounded-lg"
            style={{
              background: "#757575",
              color: "#E0E0E0",
            }}
          >
            Back to CNN Upload
          </button>
        </div>
      </div>
    );
  }

  if (!resultData) {
    return (
      <div className="min-h-screen bg-[#000000] flex items-center justify-center">
        <div className="text-[#B0B0B0] text-lg">Loading CNN results...</div>
      </div>
    );
  }

  return <CNNResultsView resultData={resultData} />;
}
