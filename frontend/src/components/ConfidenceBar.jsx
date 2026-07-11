function ConfidenceBar({ value, compact = false }) {
  const percent = Math.round((value || 0) * 100);

  return (
    <div className={`confidence ${compact ? "compact" : ""}`}>
      <div className="confidence-header">
        <span>Confidence</span>
        <strong>{percent}%</strong>
      </div>
      <div className="confidence-track" aria-label={`Confidence ${percent}%`}>
        <div className="confidence-fill" style={{ "--confidence": `${percent}%` }} />
      </div>
    </div>
  );
}

export default ConfidenceBar;
