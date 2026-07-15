import { useState } from "react";

import { sendFeedback } from "../services/api.js";

const formatClass = (value = "") => value.replaceAll("___", " · ").replaceAll("__", " ").replaceAll("_", " ");

function DetectionResultCard({ result, isLoading }) {
  const [feedback, setFeedback] = useState("");
  if (isLoading) return <article className="card detection-result-card" role="status"><h2 className="card-title">Detection result</h2><div className="skeleton"></div></article>;
  if (!result) return <article className="card detection-result-card empty-card"><h2 className="card-title">Detection result</h2><p>No result yet. Select and analyze a valid leaf image.</p></article>;

  const submit = async (message) => {
    setFeedback("Sending…");
    try {
      await sendFeedback({ scan_id: result.scan_id, message });
      setFeedback("Feedback saved.");
    } catch (error) {
      setFeedback(error.message);
    }
  };

  return (
    <article className="card detection-result-card">
      <h2 className="card-title">Detection result</h2>
      <div className="result-row"><span className="result-label">Prediction</span><strong className="result-value">{result.disease_name || formatClass(result.class_name)}</strong></div>
      <div className="result-row"><span className="result-label">Confidence</span><strong className="result-value">{(result.confidence * 100).toFixed(2)}%</strong></div>
      <div className="result-row"><span className="result-label">Crop</span><strong className="result-value">{result.crop || "Unavailable"}</strong></div>
      <div className="result-row"><span className="result-label">Status</span><strong className="result-value">{result.detection_status.replaceAll("_", " ")}</strong></div>
      <div className="result-row"><span className="result-label">Model</span><strong className="result-value">{result.model_name} {result.model_version}</strong></div>
      {result.quality_warnings.length > 0 && <div className="quality-warning"><strong>Photo quality review</strong><ul>{result.quality_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
      <details><summary>Top model alternatives</summary><ol className="alternatives">{result.top_3_predictions.map((item) => <li key={item.class_name}><span>{formatClass(item.class_name)}</span><strong>{(item.confidence * 100).toFixed(2)}%</strong></li>)}</ol></details>
      <div className="feedback-actions"><span>Was this result useful?</span><button type="button" onClick={() => submit("helpful")}>Yes</button><button type="button" onClick={() => submit("not_helpful")}>No</button>{feedback && <small role="status">{feedback}</small>}</div>
    </article>
  );
}

export default DetectionResultCard;
