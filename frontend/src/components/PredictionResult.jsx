import { useState } from "react";

import { sendFeedback } from "../services/api.js";
import ConfidenceBar from "./ConfidenceBar.jsx";
import DiseaseInfoCard from "./DiseaseInfoCard.jsx";

function PredictionResult({ result, previewUrl }) {
  const [open, setOpen] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");

  if (!result) {
    return (
      <section className="result-panel empty-state">
        <p>No scan yet. Upload a leaf image to view the prediction.</p>
      </section>
    );
  }

  const submitFeedback = async () => {
    setFeedbackStatus("Sending...");
    try {
      await sendFeedback({ predicted_class: result.class_name, confidence: result.confidence, message: "User marked result as wrong" });
      setFeedbackStatus("Feedback logged.");
    } catch (error) {
      setFeedbackStatus(error.message);
    }
  };

  return (
    <section className="result-panel">
      {previewUrl && <img className="preview-image" src={previewUrl} alt="Uploaded leaf preview" />}
      <div className="result-content">
        <p className="eyebrow">{result.mode === "mock" ? "Mock fallback" : "Model prediction"}</p>
        <h2>{result.class_name.replaceAll("_", " ")}</h2>
        <ConfidenceBar value={result.confidence} />
        <button className="link-button" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide alternatives" : "Show top 3 alternatives"}
        </button>
        {open && (
          <ol className="alternatives">
            {(result.top_3_predictions || []).map((item) => (
              <li key={item.class_name}>
                <span>{item.class_name.replaceAll("_", " ")}</span>
                <strong>{Math.round(item.confidence * 100)}%</strong>
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
          <button className="secondary-button" onClick={submitFeedback}>This seems wrong</button>
          {feedbackStatus && <span>{feedbackStatus}</span>}
        </div>
      </div>
    </section>
  );
}

export default PredictionResult;
