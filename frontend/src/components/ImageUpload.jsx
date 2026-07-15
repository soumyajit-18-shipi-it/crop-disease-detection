import { useRef, useState } from "react";

const MAX_SIZE_MB = 10;

function LeafIcon() {
  return (
    <svg className="leaf-icon" viewBox="0 0 64 64" role="img" aria-label="Leaf">
      <path d="M53 10C31 11 15 22 12 39c-2 11 6 18 16 15 17-4 25-21 25-44Z" />
      <path d="M17 48c9-13 19-22 31-30" />
    </svg>
  );
}

function ImageUpload({ onImageSelected, isLoading }) {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewName, setPreviewName] = useState("");
  const [error, setError] = useState("");

  const validate = (file) => {
    if (!file) return "No file selected.";
    if (!file.type.startsWith("image/")) return "Please choose a JPG or PNG leaf image.";
    if (file.size > MAX_SIZE_MB * 1024 * 1024) return `Image must be ${MAX_SIZE_MB}MB or smaller.`;
    return "";
  };

  const handleFile = (file) => {
    const validationError = validate(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError("");
    setPreviewName(file.name);
    onImageSelected(file);
  };

  return (
    <section
      className={`upload-panel ${dragActive ? "drag-active" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragActive(false);
        handleFile(event.dataTransfer.files?.[0]);
      }}
    >
      <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => handleFile(event.target.files?.[0])} hidden />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(event) => handleFile(event.target.files?.[0])}
        hidden
      />
      <div className="drop-zone" onClick={() => inputRef.current?.click()} role="button" tabIndex={0}>
        <LeafIcon />
        <div>
          <p className="eyebrow"><span></span>Upload image</p>
          <h2>Drop leaf image here</h2>
          <p>JPG or PNG up to 10MB. Natural light, leaf flat, symptomatic area centered.</p>
          {previewName && <p className="file-readout">{previewName}</p>}
          {error && <p className="inline-error">{error}</p>}
        </div>
      </div>
      <div className="upload-actions">
        <button className="primary-button" disabled={isLoading} onClick={() => inputRef.current?.click()}>
          {isLoading ? "Analyzing" : "Choose Image"}
        </button>
        <button className="secondary-button" disabled={isLoading} onClick={() => cameraRef.current?.click()}>
          Use Camera
        </button>
      </div>
    </section>
  );
}

export default ImageUpload;
