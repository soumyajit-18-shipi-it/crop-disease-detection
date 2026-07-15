const formatClass = (value = "") => value.replaceAll("___", " · ").replaceAll("__", " ").replaceAll("_", " ");

function DetectionHistoryCard({ scans, expanded = false, onViewAll }) {
  return (
    <article className={`card history-card${expanded ? " expanded-history" : ""}`}>
      <div className="card-heading-row"><h2 className="card-title">{expanded ? "All matching scans" : "Recent scans"}</h2>{onViewAll && scans.length > 0 && <button type="button" onClick={onViewAll}>View all</button>}</div>
      {scans.length === 0 ? <p className="empty-copy">No scan records match this view.</p> : <ul>{scans.map((scan) => <li className="history-item" key={scan.id}><div><strong>{formatClass(scan.predicted_class)}</strong><span>{(scan.confidence * 100).toFixed(2)}% · {scan.detection_status?.replaceAll("_", " ") || "recorded"}</span></div><div><time>{new Date(scan.timestamp).toLocaleString()}</time><small>{scan.model_name ? `${scan.model_name} ${scan.model_version || ""}` : "Model metadata unavailable"}</small></div></li>)}</ul>}
    </article>
  );
}

export default DetectionHistoryCard;
