import ConfidenceBar from "./ConfidenceBar.jsx";
import DiseaseInfoCard from "./DiseaseInfoCard.jsx";

function PredictionResult({ result, previewUrl }) {
  if (!result) {
    return (
      <section className="result-panel empty-state">
        <p>No scan yet. Upload a leaf image to view the mock prediction.</p>
      </section>
    );
  }

  return (
    <section className="result-panel">
      {previewUrl && <img className="preview-image" src={previewUrl} alt="Uploaded leaf preview" />}
      <div className="result-content">
        <p className="eyebrow">Mock prediction</p>
        <h2>{result.class_name.replaceAll("_", " ")}</h2>
        <ConfidenceBar value={result.confidence} />
        <DiseaseInfoCard symptoms={result.symptoms} treatment={result.recommended_treatment} />
      </div>
    </section>
  );
}

export default PredictionResult;
