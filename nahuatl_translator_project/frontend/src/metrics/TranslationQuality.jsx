import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, PieChart, Pie,
} from "recharts";
import MetricCard from "./MetricCard";
import ScoreTable from "./ScoreTable";

const COLORS = ["#2a6bff", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"];

export default function TranslationQuality({ evalResult }) {
  if (!evalResult) return <p style={{ opacity: 0.6 }}>Run an evaluation to see translation quality metrics.</p>;

  const { aggregate, by_category, by_direction, pairs } = evalResult;

  // Category chart data
  const categoryData = Object.entries(by_category || {}).map(([name, scores]) => ({
    name,
    BLEU: +(scores.bleu || 0).toFixed(3),
    chrF: +(scores.chrf || 0).toFixed(3),
    METEOR: +(scores.meteor || 0).toFixed(3),
    count: scores.count || 0,
  }));

  // Direction chart data
  const directionData = Object.entries(by_direction || {}).map(([name, scores]) => ({
    name,
    BLEU: +(scores.bleu || 0).toFixed(3),
    chrF: +(scores.chrf || 0).toFixed(3),
    METEOR: +(scores.meteor || 0).toFixed(3),
    TER: +(scores.ter || 0).toFixed(3),
    count: scores.count || 0,
  }));

  // Spanish contamination data
  const contaminated = (pairs || []).filter((p) => p.spanish_words_found && p.spanish_words_found.length > 0);
  const contaminationPie = [
    { name: "Clean", value: (pairs || []).length - contaminated.length },
    { name: "Contaminated", value: contaminated.length },
  ];

  return (
    <div>
      {/* Aggregate score cards */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <MetricCard label="BLEU" value={aggregate?.bleu} metric="bleu" subtitle="Higher is better" />
        <MetricCard label="chrF" value={aggregate?.chrf} metric="chrf" subtitle="Higher is better" />
        <MetricCard label="METEOR" value={aggregate?.meteor} metric="meteor" subtitle="Higher is better" />
        <MetricCard label="TER" value={aggregate?.ter} metric="ter" subtitle="Lower is better" />
      </div>

      {/* Category breakdown */}
      {categoryData.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Scores by Category</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="name" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }}
                labelStyle={{ color: "#e9eef5" }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="BLEU" fill="#2a6bff" radius={[4, 4, 0, 0]} />
              <Bar dataKey="chrF" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="METEOR" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Direction breakdown */}
      {directionData.length > 0 && (
        <div style={{ marginBottom: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Scores by Direction</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={directionData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="name" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }}
                labelStyle={{ color: "#e9eef5" }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="BLEU" fill="#2a6bff" radius={[4, 4, 0, 0]} />
              <Bar dataKey="chrF" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="METEOR" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              <Bar dataKey="TER" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Spanish contamination */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 24 }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Spanish Contamination</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={contaminationPie}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={2}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                <Cell fill="#22c55e" />
                <Cell fill="#ef4444" />
              </Pie>
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
        {contaminated.length > 0 && (
          <div style={{ flex: 1, minWidth: 200 }}>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>Contaminated Pairs</h3>
            <div style={{ maxHeight: 180, overflowY: "auto", fontSize: 12 }}>
              {contaminated.map((p, i) => (
                <div key={i} style={{ marginBottom: 6, padding: 6, background: "#0f1422", borderRadius: 6 }}>
                  <div style={{ opacity: 0.7 }}>{p.src}</div>
                  <div style={{ color: "#ef4444" }}>Spanish: {p.spanish_words_found.join(", ")}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Per-pair detail table */}
      <h3 style={{ fontSize: 14, marginBottom: 8 }}>Per-Pair Results</h3>
      <ScoreTable pairs={pairs} />
    </div>
  );
}
