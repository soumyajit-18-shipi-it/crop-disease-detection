import { useEffect, useState } from "react";

import { getHistory } from "../services/api.js";

function ScanHistory({ refreshKey = 0 }) {
  const [scans, setScans] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    getHistory(5)
      .then((items) => {
        if (mounted) setScans(items);
      })
      .catch(() => {
        if (mounted) setError("Recent scans unavailable");
      });
    return () => {
      mounted = false;
    };
  }, [refreshKey]);

  return (
    <section className="history-panel">
      <p className="eyebrow"><span></span>Recent scans</p>
      <h2>Field log</h2>
      {error ? (
        <p className="inline-error">{error}</p>
      ) : scans.length === 0 ? (
        <div className="mini-empty">
          <span className="leaf-glyph">LL</span>
          <p>No backend scan history yet.</p>
        </div>
      ) : (
        <ul>
          {scans.map((scan) => (
            <li key={scan.id}>
              <span>{scan.predicted_class.replaceAll("_", " ")}</span>
              <strong>{Math.round(scan.confidence * 100)}%</strong>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default ScanHistory;
