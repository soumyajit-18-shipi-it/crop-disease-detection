function severityClass(severity = "") {
  const normalized = severity.toLowerCase();
  if (normalized.includes("severe")) return "severity severe";
  if (normalized.includes("moderate")) return "severity moderate";
  if (normalized.includes("mild") || normalized.includes("healthy")) return "severity mild";
  return "severity review";
}

function DiseaseInfoCard({ crop, diseaseName, severity, symptoms, treatment }) {
  return (
    <div className="info-grid">
      <article className="disease-profile">
        <div className="chip-row">
          {crop && <span className="chip">{crop}</span>}
          <span className={severityClass(severity)}>{severity || "needs expert review"}</span>
        </div>
        <h3>{diseaseName || "Disease profile"}</h3>
        <p>{symptoms}</p>
      </article>
      <article className="treatment-card">
        <p className="eyebrow"><span></span>Advice</p>
        <h3>Recommended treatment</h3>
        <p>{treatment}</p>
      </article>
    </div>
  );
}

export default DiseaseInfoCard;
