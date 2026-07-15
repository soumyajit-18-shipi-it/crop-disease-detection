import { useEffect, useMemo, useState } from "react";

import ImageUpload from "../components/ImageUpload.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import ScanHistory from "../components/ScanHistory.jsx";
import { getHealth, predictDisease } from "../services/api.js";

function Home() {
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState("checking");

  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : ""), [selectedFile]);

  useEffect(() => {
    let mounted = true;
    const checkBackend = async () => {
      try {
        const health = await getHealth();
        if (mounted) setBackendStatus(health.status === "ok" ? "online" : "degraded");
      } catch {
        if (mounted) setBackendStatus("offline");
      }
    };
    checkBackend();
    const timer = window.setInterval(checkBackend, 15000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  const handleImageSelected = async (file) => {
    setSelectedFile(file);
    setIsLoading(true);
    setError("");
    try {
      const prediction = await predictDisease(file);
      setResult({ ...prediction, scanned_at: new Date().toISOString() });
      setHistoryRefreshKey((value) => value + 1);
    } catch {
      setError("We could not analyze this image. Check your connection and try another clear leaf photo.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="home-layout">
      <section className="section-heading">
        <p className="eyebrow"><span></span>Crop scanner</p>
        <h1>Inspect a leaf image with field-ready model feedback.</h1>
        <p>Use natural light, keep the leaf flat, and center the symptomatic area for the clearest result.</p>
      </section>

      <ImageUpload onImageSelected={handleImageSelected} isLoading={isLoading} />
      {error && <div className="error-banner">{error}</div>}

      <div className="content-grid">
        <PredictionResult
          result={result}
          previewUrl={previewUrl}
          isLoading={isLoading}
          backendStatus={backendStatus}
        />
        <ScanHistory refreshKey={historyRefreshKey} />
      </div>
    </div>
  );
}

export default Home;
