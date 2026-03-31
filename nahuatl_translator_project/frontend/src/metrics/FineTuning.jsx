import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";

export default function FineTuning() {
  const [logData, setLogData] = useState(null);
  const [error, setError] = useState("");

  function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        setLogData(Array.isArray(data) ? data : data.epochs || data.log || [data]);
        setError("");
      } catch (err) {
        setError("Invalid JSON file");
      }
    };
    reader.readAsText(file);
  }

  if (!logData) {
    return (
      <div>
        <p style={{ opacity: 0.6, marginBottom: 12 }}>
          Upload a <code>training_log.json</code> file from a fine-tuning run to visualize training metrics.
        </p>
        <p style={{ opacity: 0.5, fontSize: 12, marginBottom: 12 }}>
          Expected format: array of objects with fields like <code>epoch</code>, <code>train_loss</code>,{" "}
          <code>val_loss</code>, <code>learning_rate</code>, <code>bleu</code>, <code>chrf</code>.
        </p>
        <input type="file" accept=".json" onChange={handleFileUpload} />
        {error && <p style={{ color: "#ff7a7a", marginTop: 8 }}>{error}</p>}
      </div>
    );
  }

  const hasLoss = logData.some((d) => d.train_loss != null || d.val_loss != null);
  const hasLR = logData.some((d) => d.learning_rate != null);
  const hasMetrics = logData.some((d) => d.bleu != null || d.chrf != null);

  // Detect overfitting: val_loss increasing for 2+ consecutive epochs
  let overfitEpoch = null;
  if (hasLoss) {
    let increasing = 0;
    for (let i = 1; i < logData.length; i++) {
      if (logData[i].val_loss != null && logData[i - 1].val_loss != null) {
        if (logData[i].val_loss > logData[i - 1].val_loss) {
          increasing++;
          if (increasing >= 2 && overfitEpoch === null) {
            overfitEpoch = logData[i].epoch || i;
          }
        } else {
          increasing = 0;
        }
      }
    }
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontSize: 14, margin: 0 }}>Fine-tuning Metrics ({logData.length} epochs)</h3>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {overfitEpoch !== null && (
            <span
              style={{
                background: "#ef444433",
                color: "#ef4444",
                padding: "4px 10px",
                borderRadius: 8,
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              Overfitting detected at epoch {overfitEpoch}
            </span>
          )}
          <button
            className="secondary"
            onClick={() => setLogData(null)}
            style={{ width: "auto", padding: "6px 12px", fontSize: 12 }}
          >
            Load different file
          </button>
        </div>
      </div>

      {/* Loss curves */}
      {hasLoss && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, marginBottom: 8 }}>Training & Validation Loss</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={logData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="epoch" tick={{ fill: "#e9eef5", fontSize: 11 }} label={{ value: "Epoch", fill: "#e9eef5", fontSize: 11, position: "bottom" }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="train_loss" stroke="#2a6bff" strokeWidth={2} dot={{ r: 3 }} name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Val Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Learning rate */}
      {hasLR && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, marginBottom: 8 }}>Learning Rate Schedule</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={logData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="epoch" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} tickFormatter={(v) => v.toExponential(1)} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Line type="monotone" dataKey="learning_rate" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} name="LR" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* BLEU/chrF progression */}
      {hasMetrics && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 13, marginBottom: 8 }}>Validation Metrics per Epoch</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={logData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="epoch" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="bleu" stroke="#2a6bff" strokeWidth={2} dot={{ r: 3 }} name="BLEU" />
              <Line type="monotone" dataKey="chrf" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} name="chrF" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
