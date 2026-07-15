function FieldOverviewCard({ summary }) {
  return (
    <article className="card field-overview-card">
      <h2 className="card-title">Scan overview</h2>
      <dl className="overview-list">
        <div><dt>Healthy classifications</dt><dd>{summary.healthy_percentage == null ? "—" : `${summary.healthy_percentage.toFixed(1)}%`}</dd></div>
        <div><dt>Disease classifications</dt><dd>{summary.diseased_scans}</dd></div>
        <div><dt>Distinct disease classes</dt><dd>{summary.active_disease_classes}</dd></div>
        <div><dt>Latest scan</dt><dd>{summary.latest_scan_at ? new Date(summary.latest_scan_at).toLocaleString() : "—"}</dd></div>
      </dl>
    </article>
  );
}

export default FieldOverviewCard;
