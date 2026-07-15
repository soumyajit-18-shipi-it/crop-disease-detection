import { useEffect, useMemo, useState } from "react";

import DetectionResultCard from "../components/DetectionResultCard.jsx";
import RecentDetectionCard from "../components/RecentDetectionCard.jsx";
import TreatmentCard from "../components/TreatmentCard.jsx";
import UploadCard from "../components/UploadCard.jsx";
import { getHealth, predictDisease } from "../services/api.js";

function Scan() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => {
    let active = true;
    getHealth().then((value) => active && setHealth(value)).catch(() => active && setHealth(null));
    return () => { active = false; };
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError("");
    try {
      setResult(await predictDisease(file));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="main-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Model-backed analysis</p>
          <h1>Analyze a crop leaf</h1>
          <p>
            {health?.model_loaded
              ? `${health.model_name} ${health.model_version} · ${health.input_size.width}×${health.input_size.height} RGB`
              : "The API must report a healthy, loaded model before inference can succeed."}
          </p>
        </div>
      </div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      <section className="scan-grid">
        <UploadCard selectedFile={file} onFileSelected={setFile} onAnalyze={analyze} isLoading={loading} />
        <RecentDetectionCard previewUrl={previewUrl} result={result} isLoading={loading} />
        <DetectionResultCard result={result} isLoading={loading} />
        <TreatmentCard result={result} isLoading={loading} />
      </section>
    </main>
  );
}

export default Scan;
