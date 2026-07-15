function RecentDetectionCard({ previewUrl, result, isLoading }) {
  return (
    <article className="card recent-detection-card">
      <h2 className="card-title">Image under analysis</h2>
      {previewUrl ? <div className="detection-image-container"><img src={previewUrl} alt="Selected crop leaf" /></div> : <div className="image-empty">No image selected.</div>}
      {isLoading && <p className="loading-copy" role="status">Running the released ONNX model…</p>}
      {result && <p className="scan-record-meta">Saved as scan #{result.scan_id} at {new Date(result.scanned_at).toLocaleString()}.</p>}
    </article>
  );
}

export default RecentDetectionCard;
