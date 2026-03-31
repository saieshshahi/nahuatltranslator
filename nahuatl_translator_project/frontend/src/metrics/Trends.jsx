import React, { useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";

export default function Trends({ history, fetchHistory }) {
  useEffect(() => {
    if (history.length === 0) fetchHistory();
  }, [history, fetchHistory]);

  if (history.length === 0) {
    return (
      <p style={{ opacity: 0.6 }}>
        No evaluation history yet. Run at least 2 evaluations to see trends over time.
      </p>
    );
  }

  // Sort oldest first for the chart
  const sorted = [...history].reverse();

  const trendData = sorted.map((h, i) => ({
    run: i + 1,
    label: h.timestamp ? new Date(h.timestamp).toLocaleDateString() : `Run ${i + 1}`,
    BLEU: +(h.aggregate?.bleu || 0).toFixed(3),
    chrF: +(h.aggregate?.chrf || 0).toFixed(3),
    METEOR: +(h.aggregate?.meteor || 0).toFixed(3),
    TER: +(h.aggregate?.ter || 0).toFixed(3),
    latency: +(h.avg_latency_ms || 0).toFixed(0),
    cost: +(h.total_cost_estimate || 0).toFixed(5),
    pairs: h.pair_count || 0,
    model: h.model || "unknown",
  }));

  // Regression detection
  let regression = null;
  if (trendData.length >= 2) {
    const last = trendData[trendData.length - 1];
    const prev = trendData[trendData.length - 2];
    if (last.BLEU < prev.BLEU - 0.05 || last.chrF < prev.chrF - 0.05) {
      regression = { bleu: last.BLEU - prev.BLEU, chrf: last.chrF - prev.chrF };
    }
  }

  return (
    <div>
      {regression && (
        <div
          style={{
            background: "#ef444422",
            border: "1px solid #ef4444",
            borderRadius: 10,
            padding: "10px 16px",
            marginBottom: 16,
            fontSize: 13,
          }}
        >
          <strong>Regression detected:</strong> BLEU dropped by {Math.abs(regression.bleu).toFixed(3)},
          chrF dropped by {Math.abs(regression.chrf).toFixed(3)} compared to previous run.
        </div>
      )}

      <h3 style={{ fontSize: 14, marginBottom: 8 }}>Score Trends ({trendData.length} runs)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={trendData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
          <XAxis dataKey="label" tick={{ fill: "#e9eef5", fontSize: 10 }} />
          <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="BLEU" stroke="#2a6bff" strokeWidth={2} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="chrF" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="METEOR" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="TER" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>

      {/* Latency trend */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 8 }}>Latency Trend</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={trendData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
            <XAxis dataKey="label" tick={{ fill: "#e9eef5", fontSize: 10 }} />
            <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
            <Line type="monotone" dataKey="latency" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} name="Avg Latency (ms)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* History table */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 8 }}>Evaluation History</h3>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, background: "#0f1422", borderRadius: 10, overflow: "hidden" }}>
            <thead>
              <tr style={{ background: "#1b243d", textAlign: "left" }}>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Model</th>
                <th style={thStyle}>Pairs</th>
                <th style={thStyle}>BLEU</th>
                <th style={thStyle}>chrF</th>
                <th style={thStyle}>Latency</th>
                <th style={thStyle}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((h, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #202841" }}>
                  <td style={tdStyle}>{h.timestamp ? new Date(h.timestamp).toLocaleString() : "—"}</td>
                  <td style={tdStyle}>{h.model || "—"}</td>
                  <td style={tdStyle}>{h.pair_count || "—"}</td>
                  <td style={tdStyle}>{(h.aggregate?.bleu || 0).toFixed(3)}</td>
                  <td style={tdStyle}>{(h.aggregate?.chrf || 0).toFixed(3)}</td>
                  <td style={tdStyle}>{Math.round(h.avg_latency_ms || 0)}ms</td>
                  <td style={tdStyle}>${(h.total_cost_estimate || 0).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const thStyle = { padding: "8px 6px", fontSize: 11, opacity: 0.85, whiteSpace: "nowrap" };
const tdStyle = { padding: "6px", whiteSpace: "nowrap" };
