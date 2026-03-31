import React, { useState } from "react";

function cellColor(value, metric) {
  if (metric === "ter") {
    if (value <= 0.3) return "#22c55e33";
    if (value <= 0.6) return "#f59e0b33";
    return "#ef444433";
  }
  if (value >= 0.7) return "#22c55e33";
  if (value >= 0.4) return "#f59e0b33";
  return "#ef444433";
}

export default function ScoreTable({ pairs }) {
  const [sortKey, setSortKey] = useState("chrf");
  const [sortAsc, setSortAsc] = useState(false);

  if (!pairs || pairs.length === 0) return null;

  const sorted = [...pairs].sort((a, b) => {
    const av = a[sortKey] ?? 0;
    const bv = b[sortKey] ?? 0;
    return sortAsc ? av - bv : bv - av;
  });

  function handleSort(key) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  }

  const arrow = (key) => sortKey === key ? (sortAsc ? " \u25B2" : " \u25BC") : "";

  return (
    <div style={{ overflowX: "auto", marginTop: 12 }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
          background: "#0f1422",
          borderRadius: 10,
          overflow: "hidden",
        }}
      >
        <thead>
          <tr style={{ background: "#1b243d", textAlign: "left" }}>
            <th style={thStyle}>#</th>
            <th style={thStyle}>Source</th>
            <th style={thStyle}>Expected</th>
            <th style={thStyle}>Predicted</th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("category")}>
              Cat{arrow("category")}
            </th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("bleu")}>
              BLEU{arrow("bleu")}
            </th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("chrf")}>
              chrF{arrow("chrf")}
            </th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("meteor")}>
              METEOR{arrow("meteor")}
            </th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("ter")}>
              TER{arrow("ter")}
            </th>
            <th style={{ ...thStyle, cursor: "pointer" }} onClick={() => handleSort("latency_ms")}>
              ms{arrow("latency_ms")}
            </th>
            <th style={thStyle}>Spanish</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #202841" }}>
              <td style={tdStyle}>{i + 1}</td>
              <td style={{ ...tdStyle, maxWidth: 160 }} title={p.src}>
                {truncate(p.src, 40)}
              </td>
              <td style={{ ...tdStyle, maxWidth: 160 }} title={p.tgt_expected}>
                {truncate(p.tgt_expected, 40)}
              </td>
              <td style={{ ...tdStyle, maxWidth: 160 }} title={p.tgt_predicted}>
                {truncate(p.tgt_predicted, 40)}
              </td>
              <td style={tdStyle}>{p.category}</td>
              <td style={{ ...tdStyle, background: cellColor(p.bleu, "bleu") }}>
                {p.bleu.toFixed(3)}
              </td>
              <td style={{ ...tdStyle, background: cellColor(p.chrf, "chrf") }}>
                {p.chrf.toFixed(3)}
              </td>
              <td style={{ ...tdStyle, background: cellColor(p.meteor, "meteor") }}>
                {p.meteor.toFixed(3)}
              </td>
              <td style={{ ...tdStyle, background: cellColor(p.ter, "ter") }}>
                {p.ter.toFixed(3)}
              </td>
              <td style={tdStyle}>{Math.round(p.latency_ms)}</td>
              <td style={tdStyle}>
                {p.spanish_words_found.length > 0 ? (
                  <span style={{ color: "#ef4444" }}>{p.spanish_words_found.join(", ")}</span>
                ) : (
                  <span style={{ color: "#22c55e" }}>clean</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle = { padding: "8px 6px", fontSize: 11, opacity: 0.85, whiteSpace: "nowrap" };
const tdStyle = { padding: "6px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" };

function truncate(s, n) {
  return s && s.length > n ? s.slice(0, n) + "…" : s;
}
