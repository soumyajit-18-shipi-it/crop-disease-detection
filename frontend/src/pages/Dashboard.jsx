import { useEffect, useState } from "react";

import DetectionHistoryCard from "../components/DetectionHistoryCard.jsx";
import DiseaseDistributionCard from "../components/DiseaseDistributionCard.jsx";
import FieldOverviewCard from "../components/FieldOverviewCard.jsx";
import { getDashboard } from "../services/api.js";

function Dashboard({ user, onNavigate }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    getDashboard()
      .then((value) => active && setSummary(value))
      .catch((requestError) => active && setError(requestError.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [refreshKey]);

  return (
    <main className="main-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Private workspace</p>
          <h1>Welcome, {user.name}</h1>
          <p>Your dashboard is calculated from your saved Leaflight scans.</p>
        </div>
        <button className="primary-button" type="button" onClick={() => onNavigate("scan")}>Analyze a leaf</button>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error} <button type="button" onClick={() => setRefreshKey((value) => value + 1)}>Retry</button>
        </div>
      )}
      {loading ? (
        <div className="dashboard-loading" role="status">Loading your scan statistics…</div>
      ) : summary ? (
        <>
          <section className="stats-grid" aria-label="Scan statistics">
            <article className="stat-card"><span>Total scans</span><strong>{summary.total_scans}</strong></article>
            <article className="stat-card"><span>Average confidence</span><strong>{summary.average_confidence == null ? "—" : `${(summary.average_confidence * 100).toFixed(1)}%`}</strong></article>
            <article className="stat-card"><span>Healthy scans</span><strong>{summary.healthy_scans}</strong></article>
            <article className="stat-card"><span>Needs another photo</span><strong>{summary.low_confidence_scans}</strong></article>
          </section>
          {summary.total_scans === 0 ? (
            <section className="empty-card">
              <h2>No scans yet</h2>
              <p>Analyze your first leaf image to populate statistics, disease distribution, and history.</p>
              <button className="primary-button" type="button" onClick={() => onNavigate("scan")}>Start first scan</button>
            </section>
          ) : (
            <section className="dashboard-grid analytics-grid">
              <FieldOverviewCard summary={summary} />
              <DiseaseDistributionCard distribution={summary.disease_distribution} />
              <DetectionHistoryCard scans={summary.recent_scans} onViewAll={() => onNavigate("history")} />
            </section>
          )}
        </>
      ) : null}
    </main>
  );
}

export default Dashboard;
