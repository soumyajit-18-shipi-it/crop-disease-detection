function TreatmentCard({ result, isLoading }) {
  if (isLoading) return <article className="card treatment-card" role="status"><h2 className="card-title">Reviewed guidance</h2><div className="skeleton"></div></article>;
  if (!result) return <article className="card treatment-card empty-card"><h2 className="card-title">Reviewed guidance</h2><p>Guidance appears only after a completed scan.</p></article>;
  if (result.information_status !== "reviewed" || !result.recommended_treatment) {
    return <article className="card treatment-card empty-card"><h2 className="card-title">Reviewed guidance</h2><p>No reviewed treatment guidance is stored for this model class. Consult a qualified local agronomist before treatment decisions.</p></article>;
  }
  return (
    <article className="card treatment-card">
      <h2 className="card-title">Reviewed guidance</h2>
      <div><h3>Observed symptoms</h3><p>{result.symptoms}</p></div>
      <div><h3>Recommended next steps</h3><p>{result.recommended_treatment}</p></div>
      {result.severity_level && <p className="severity-readout">Guidance severity: {result.severity_level}</p>}
    </article>
  );
}

export default TreatmentCard;
