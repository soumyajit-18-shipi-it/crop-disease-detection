import { useRef, useState } from "react";

const MAX_SIZE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

function UploadCard({ selectedFile, onFileSelected, onAnalyze, isLoading }) {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");

  const acceptFile = (file) => {
    if (!file) return;
    if (!ALLOWED_TYPES.has(file.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size === 0 || file.size > MAX_SIZE_BYTES) {
      setError("Image must be non-empty and no larger than 10 MB.");
      return;
    }
    setError("");
    onFileSelected(file);
  };

  return (
    <article className="card upload-card">
      <h2 className="card-title">Choose a leaf image</h2>
      <div className={`upload-dropzone${dragActive ? " drag-active" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={(event) => { event.preventDefault(); setDragActive(false); acceptFile(event.dataTransfer.files?.[0]); }}>
        <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => acceptFile(event.target.files?.[0])} />
        <input ref={cameraRef} type="file" accept="image/*" capture="environment" onChange={(event) => acceptFile(event.target.files?.[0])} />
        <p>Drop a focused leaf photo here, or choose a source.</p>
        <p className="upload-subtext">JPEG, PNG, or WebP · maximum 10 MB</p>
        {selectedFile && <p className="file-name-preview">{selectedFile.name} · {(selectedFile.size / 1024).toFixed(1)} KB</p>}
        {error && <p className="upload-error" role="alert">{error}</p>}
        <div className="upload-source-actions"><button type="button" onClick={() => inputRef.current?.click()} disabled={isLoading}>Choose image</button><button type="button" onClick={() => cameraRef.current?.click()} disabled={isLoading}>Use camera</button></div>
      </div>
      <button className="analyze-btn" type="button" disabled={isLoading || !selectedFile} onClick={onAnalyze}>{isLoading ? "Analyzing…" : "Analyze securely"}</button>
    </article>
  );
}

export default UploadCard;
