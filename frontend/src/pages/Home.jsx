import { useMemo, useState } from "react";

import ImageUpload from "../components/ImageUpload.jsx";
import PredictionResult from "../components/PredictionResult.jsx";
import ScanHistory from "../components/ScanHistory.jsx";
import { predictDisease } from "../services/api.js";

function Home() {
  const [result, setResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : ""), [selectedFile]);

  const handleImageSelected = async (file) => {
    setSelectedFile(file);
    setIsLoading(true);
    setError("");
    try {
      const prediction = await predictDisease(file);
      setResult(prediction);
      setHistory((current) => [prediction, ...current].slice(0, 5));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="home-layout">
      <ImageUpload onImageSelected={handleImageSelected} isLoading={isLoading} />
      {error && <div className="error-banner">{error}</div>}
      <div className="content-grid">
        {isLoading ? <section className="result-panel skeleton" /> : <PredictionResult result={result} previewUrl={previewUrl} />}
        <ScanHistory scans={history} />
      </div>
    </div>
  );
}

export default Home;
