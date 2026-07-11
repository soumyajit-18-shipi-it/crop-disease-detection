import { useRef, useState } from "react";

const MAX_SIZE_MB = 10;

function ImageUpload({ onImageSelected, isLoading }) {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [error, setError] = useState("");

  const validate = (file) => {
    if (!file) return "No file selected.";
    if (!file.type.startsWith("image/")) return "Please choose an image file.";
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
    setPreviewUrl(URL.createObjectURL(file));
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
      <input ref={inputRef} type="file" accept="image/*" onChange={(event) => handleFile(event.target.files?.[0])} hidden />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(event) => handleFile(event.target.files?.[0])}
        hidden
      />
      <div className="upload-copy">
        <p className="eyebrow">Field scan</p>
        <h1>Diagnose a leaf image</h1>
        <p>Drop a crop leaf photo here, choose one from your device, or capture a fresh field image on mobile.</p>
        {error && <p className="inline-error">{error}</p>}
      </div>
      {previewUrl && <img className="upload-preview" src={previewUrl} alt="Selected leaf preview" />}
      <div className="upload-actions">
        <button className="primary-button" disabled={isLoading} onClick={() => inputRef.current?.click()}>
          {isLoading ? "Scanning..." : "Choose Image"}
        </button>
        <button className="secondary-button" disabled={isLoading} onClick={() => cameraRef.current?.click()}>
          Use Camera
        </button>
      </div>
    </section>
  );
}

export default ImageUpload;
