function DiseaseInfoCard({ crop, diseaseName, severity, symptoms, treatment }) {
  return (
    <div className="info-grid">
      <article>
        <h3>Disease Profile</h3>
        <p><strong>Crop:</strong> {crop || "Unknown"}</p>
        <p><strong>Disease:</strong> {diseaseName || "Unknown"}</p>
        <p><strong>Severity:</strong> {severity || "needs expert review"}</p>
      </article>
      <article>
        <h3>Symptoms</h3>
        <p>{symptoms}</p>
      </article>
      <article className="wide-card">
        <h3>Recommended Treatment</h3>
        <p>{treatment}</p>
      </article>
    </div>
  );
}

export default DiseaseInfoCard;
