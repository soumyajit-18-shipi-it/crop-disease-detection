import { useRef, useState } from "react";

function ImageUpload({ onImageSelected, isLoading }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = (file) => {
    if (file && file.type.startsWith("image/")) {
      onImageSelected(file);
    }
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
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(event) => handleFile(event.target.files?.[0])}
        hidden
      />
      <div>
        <h1>Leaf Disease Scan</h1>
        <p>Upload a clear plant leaf image to run a mock diagnosis through the FastAPI backend.</p>
      </div>
      <button className="primary-button" disabled={isLoading} onClick={() => inputRef.current?.click()}>
        {isLoading ? "Scanning..." : "Choose Image"}
      </button>
    </section>
  );
}

export default ImageUpload;
