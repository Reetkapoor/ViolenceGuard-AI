import { useState } from "react";
import { AlertTriangle, CheckCircle2, Film, Loader2, ShieldCheck, UploadCloud } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectFile = (selected) => {
    if (!selected) return;
    const allowed = [".mp4", ".avi", ".mov", ".mkv"];
    const valid = allowed.some((ext) => selected.name.toLowerCase().endsWith(ext));
    if (!valid) {
      setError("Upload an MP4, AVI, MOV, or MKV video.");
      return;
    }
    setFile(selected);
    setResult(null);
    setError("");
  };

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Inference failed.");
      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the FastAPI backend.");
    } finally {
      setLoading(false);
    }
  };

  const confidence = result ? Math.round(result.confidence * 100) : 0;
  const isViolence = result?.label === "Violence";

  return (
    <main className="page">
      <nav className="nav">
        <div className="brand"><ShieldCheck size={24} /> ViolenceGuard AI</div>
        <span className="status"><span /> FastAPI connected</span>
      </nav>

      <section className="hero">
        <p className="eyebrow">VIDEO MONITORING & INFERENCE</p>
        <h1>Detect violence from uploaded video.</h1>
        <p className="subtitle">
          Upload a video to trigger the MobileNetV2 + BiLSTM inference service and review
          the classification result and confidence score.
        </p>
      </section>

      <section className="grid">
        <div className="card upload-card">
          <div className="card-title"><Film size={20} /><h2>Video input</h2></div>

          <label
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              selectFile(e.dataTransfer.files?.[0]);
            }}
          >
            <UploadCloud size={34} />
            <strong>{file ? file.name : "Drop a video here"}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "or click to browse"}</span>
            <input
              type="file"
              accept=".mp4,.avi,.mov,.mkv,video/*"
              onChange={(e) => selectFile(e.target.files?.[0])}
            />
          </label>

          <button className="primary" disabled={!file || loading} onClick={runInference}>
            {loading ? <><Loader2 className="spin" size={18} /> Running inference...</> : "Run inference"}
          </button>

          {error && <div className="error"><AlertTriangle size={18} /> {error}</div>}
        </div>

        <div className="card result-card">
          <div className="card-title"><ShieldCheck size={20} /><h2>Classification result</h2></div>

          {!result && !loading && (
            <div className="empty">
              <ShieldCheck size={44} />
              <p>No inference result yet.</p>
              <span>Upload a video and run inference to see the model output.</span>
            </div>
          )}

          {loading && (
            <div className="empty">
              <Loader2 className="spin" size={44} />
              <p>Analyzing video...</p>
              <span>The FastAPI service is processing the uploaded clip.</span>
            </div>
          )}

          {result && (
            <div className="result">
              <div className={`result-badge ${isViolence ? "danger" : "safe"}`}>
                {isViolence ? <AlertTriangle size={24} /> : <CheckCircle2 size={24} />}
                <div>
                  <small>Detected class</small>
                  <strong>{result.label}</strong>
                </div>
              </div>

              <div className="confidence-row">
                <div>
                  <span>Confidence</span>
                  <strong>{confidence}%</strong>
                </div>
                <div className="bar"><span style={{ width: `${confidence}%` }} /></div>
              </div>

              <div className="meta">
                <span>Alert</span>
                <strong>{result.alert ? "Triggered" : "Not triggered"}</strong>
              </div>
              <div className="meta">
                <span>Inference timestamp</span>
                <strong>{new Date(result.timestamp).toLocaleString()}</strong>
              </div>
            </div>
          )}
        </div>
      </section>

      <footer>ViolenceGuard AI · React dashboard · FastAPI inference service</footer>
    </main>
  );
}

export default App;
