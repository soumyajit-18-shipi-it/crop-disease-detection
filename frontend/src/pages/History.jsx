import { useEffect, useState } from "react";

import DetectionHistoryCard from "../components/DetectionHistoryCard.jsx";
import { getHistory } from "../services/api.js";

function History() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    getHistory({ limit: 200, search: submittedQuery })
      .then((value) => active && setScans(value))
      .catch((requestError) => active && setError(requestError.message))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, [submittedQuery]);

  return (
    <main className="main-content">
      <div className="page-heading"><div><p className="eyebrow">Private records</p><h1>Scan history</h1><p>Only records associated with your authenticated account are returned.</p></div></div>
      <form className="history-search" onSubmit={(event) => { event.preventDefault(); setSubmittedQuery(query.trim()); }}>
        <label htmlFor="history-query">Search model class</label>
        <div><input id="history-query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search your saved prediction classes" /><button type="submit">Search</button></div>
      </form>
      {error ? <div className="error-banner" role="alert">{error}</div> : loading ? <div className="dashboard-loading" role="status">Loading scan history…</div> : <DetectionHistoryCard scans={scans} expanded />}
    </main>
  );
}

export default History;
