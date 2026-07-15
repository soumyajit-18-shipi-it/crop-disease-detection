import { useState } from "react";

import { sendFeedback } from "../services/api.js";
import ConfidenceBar from "./ConfidenceBar.jsx";
import DiseaseInfoCard from "./DiseaseInfoCard.jsx";

function formatName(name = "") {
  return name.replaceAll("_", " ");
}

function statusLabel(status, modelHealth) {
  if (modelHealth?.model_loaded === true) return "MODEL LIVE";
  if (status === "degraded") return "MODEL UNAVAILABLE";
  if (status === "offline") return "OFFLINE";
  return "CHECKING";
}

function inputLabel(inputSize) {
  const width = Number(inputSize?.width);
  const height = Number(inputSize?.height);
  const channels = Number(inputSize?.channels);
  if (width > 0 && height > 0 && channels > 0) {
    const color = channels === 3 ? "RGB" : `${channels}-channel`;
    return `${width}×${height} ${color}`;
  }
  return "RGB image";
}

function ScannerFrame({ previewUrl, backendStatus, modelHealth }) {
  return (
    <div className="scanner-frame">
      <div className="scanner-topbar">
        <span className={`live-dot ${backendStatus}`}></span>
        <span>{statusLabel(backendStatus, modelHealth)}</span>
        <span>{inputLabel(modelHealth?.input_size)}</span>
      </div>
      <div className="scanner-image-wrap">
        {previewUrl ? (
          <img className="scanner-image" src={previewUrl} alt="Leaf under analysis" />
        ) : (
          <div className="scanner-empty">Waiting for image</div>
        )}
        <div className="scanline"></div>
        <div className="bounding-box">
          <span>leaf region</span>
        </div>
      </div>
    </div>
  );
}

function PredictionResult({ result, previewUrl, isLoading, backendStatus, modelHealth }) {
  const [open, setOpen] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");

  if (isLoading) {
    return (
      <section className="result-panel scanner-result">
        <ScannerFrame previewUrl={previewUrl} backendStatus={backendStatus} modelHealth={modelHealth} />
        <div className="result-content">
          <p className="eyebrow"><span></span>Analysis running</p>
          <h2>Scanning visual symptoms</h2>
          <p className="muted">The image is being checked for leaf spots, discoloration, and texture changes.</p>
          <div className="skeleton result-skeleton"></div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="result-panel empty-state designed-empty">
        <p className="eyebrow"><span></span>No scan yet</p>
        <h2>Upload a leaf image to see diagnosis details here.</h2>
        <p>Results include top prediction, confidence, alternatives, severity, symptoms, and treatment guidance.</p>
      </section>
    );
  }

  const lowConfidence = result.confidence < 0.6;
  const timestamp = result.scanned_at ? new Date(result.scanned_at) : new Date();

  const submitFeedback = async (message) => {
    setFeedbackStatus("Sending");
    try {
      await sendFeedback({ predicted_class: result.class_name, confidence: result.confidence, message });
      setFeedbackStatus("Feedback logged");
    } catch {
      setFeedbackStatus("Could not send feedback");
    }
  };

  return (
    <section className="result-panel">
      <ScannerFrame previewUrl={previewUrl} backendStatus={backendStatus} modelHealth={modelHealth} />
      <div className="result-content">
        <div className="result-meta-row">
          <p className="eyebrow"><span></span>{result.mode === "onnx" ? "Model prediction" : "Backend fallback"}</p>
          <time>{timestamp.toLocaleString()}</time>
        </div>
        <h2>{formatName(result.class_name)}</h2>
        <ConfidenceBar value={result.confidence} />
        {lowConfidence && (
          <div className="confidence-note">
            Low confidence. Consider retaking the photo in better lighting with the affected leaf area centered.
          </div>
        )}
        <button className="link-button" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide alternatives" : "Review top 3 alternatives"}
        </button>
        {open && (
          <ol className="alternatives">
            {(result.top_3_predictions || []).map((item) => (
              <li key={item.class_name}>
                <span>{formatName(item.class_name)}</span>
                <ConfidenceBar value={item.confidence} compact />
              </li>
            ))}
          </ol>
        )}
        <DiseaseInfoCard
          crop={result.crop}
          diseaseName={result.disease_name}
          severity={result.severity_level}
          symptoms={result.symptoms}
          treatment={result.recommended_treatment}
        />
        <div className="feedback-row">
          <span>Was this helpful?</span>
          <button className="icon-feedback" onClick={() => submitFeedback("helpful")} aria-label="Helpful">+</button>
          <button className="icon-feedback" onClick={() => submitFeedback("not helpful")} aria-label="Not helpful">-</button>
          {feedbackStatus && <small>{feedbackStatus}</small>}
        </div>
      </div>
    </section>
  );
}

export default PredictionResult;
