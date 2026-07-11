function ScanHistory({ scans }) {
  return (
    <section className="history-panel">
      <h2>Recent Scans</h2>
      {scans.length === 0 ? (
        <p>No scans in this browser session.</p>
      ) : (
        <ul>
          {scans.map((scan, index) => (
            <li key={`${scan.class_name}-${index}`}>
              <span>{scan.class_name.replaceAll("_", " ")}</span>
              <strong>{Math.round(scan.confidence * 100)}%</strong>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default ScanHistory;
