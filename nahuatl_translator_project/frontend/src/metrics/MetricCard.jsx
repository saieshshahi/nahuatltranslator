import React from "react";

const COLORS = {
  excellent: "#22c55e",
  good: "#84cc16",
  fair: "#f59e0b",
  poor: "#ef4444",
};

function scoreColor(value, metric) {
  // TER is inverted (lower is better)
  if (metric === "ter") {
    if (value <= 0.3) return COLORS.excellent;
    if (value <= 0.5) return COLORS.good;
    if (value <= 0.8) return COLORS.fair;
    return COLORS.poor;
  }
  // BLEU, chrF, METEOR (higher is better)
  if (value >= 0.7) return COLORS.excellent;
  if (value >= 0.4) return COLORS.good;
  if (value >= 0.2) return COLORS.fair;
  return COLORS.poor;
}

export default function MetricCard({ label, value, metric, subtitle }) {
  const color = scoreColor(value, metric);
  const display = typeof value === "number" ? value.toFixed(3) : "—";

  return (
    <div
      style={{
        background: "#0f1422",
        border: "1px solid #2a3558",
        borderRadius: 12,
        padding: "16px 20px",
        textAlign: "center",
        minWidth: 120,
        flex: 1,
      }}
    >
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 32, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
        {display}
      </div>
      {subtitle && (
        <div style={{ fontSize: 11, opacity: 0.6, marginTop: 4 }}>{subtitle}</div>
      )}
    </div>
  );
}
