import React, { useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from "recharts";
import MetricCard from "./MetricCard";

const COLORS = ["#2a6bff", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6"];

export default function CorpusData({ corpusStats, fetchCorpusStats }) {
  useEffect(() => {
    if (!corpusStats) fetchCorpusStats();
  }, [corpusStats, fetchCorpusStats]);

  if (!corpusStats) return <p style={{ opacity: 0.6 }}>Loading corpus statistics...</p>;

  const {
    total_entries,
    en_vocab_size,
    nah_vocab_size,
    avg_en_length,
    avg_nah_length,
    category_distribution,
    contamination_rate,
    contaminated_count,
  } = corpusStats;

  // Category pie chart (golden pairs)
  const categoryPie = Object.entries(category_distribution || {}).map(([name, value]) => ({
    name,
    value,
  }));

  // Contamination visual
  const cleanCount = total_entries - contaminated_count;
  const contaminationPie = [
    { name: "Clean", value: cleanCount },
    { name: "Contaminated", value: contaminated_count },
  ];

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <MetricCard label="Corpus Entries" value={total_entries} metric="count" />
        <MetricCard label="English Vocab" value={en_vocab_size} metric="count" subtitle="unique tokens" />
        <MetricCard label="Nahuatl Vocab" value={nah_vocab_size} metric="count" subtitle="unique tokens" />
        <MetricCard label="Contamination" value={contamination_rate} metric="ter" subtitle={`${contaminated_count} entries`} />
      </div>

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <MetricCard label="Avg EN Length" value={avg_en_length} metric="count" subtitle="words/sentence" />
        <MetricCard label="Avg NAH Length" value={avg_nah_length} metric="count" subtitle="words/sentence" />
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
        {/* Category distribution */}
        {categoryPie.length > 0 && (
          <div style={{ flex: 1, minWidth: 300 }}>
            <h3 style={{ fontSize: 14, marginBottom: 8 }}>Golden Pairs by Category</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={categoryPie}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, value }) => `${name} (${value})`}
                >
                  {categoryPie.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Corpus contamination */}
        <div style={{ flex: 1, minWidth: 300 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Corpus Spanish Contamination</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={contaminationPie}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
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
      </div>

      {/* Category bar chart */}
      {categoryPie.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Golden Pairs Count by Category</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={categoryPie} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="name" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Bar dataKey="value" fill="#2a6bff" radius={[4, 4, 0, 0]} name="Pairs">
                {categoryPie.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
