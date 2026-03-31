import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell,
} from "recharts";
import MetricCard from "./MetricCard";

const COLORS = ["#2a6bff", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"];

export default function LLMPerformance({ evalResult }) {
  if (!evalResult) return <p style={{ opacity: 0.6 }}>Run an evaluation to see LLM performance metrics.</p>;

  const { pairs, by_category, by_direction, avg_latency_ms, total_cost_estimate } = evalResult;

  // Latency histogram (bucket into 10 bins)
  const latencies = (pairs || []).map((p) => p.latency_ms);
  const maxLat = Math.max(...latencies, 1);
  const binSize = Math.ceil(maxLat / 10);
  const histBins = [];
  for (let i = 0; i < 10; i++) {
    const lo = i * binSize;
    const hi = (i + 1) * binSize;
    const count = latencies.filter((l) => l >= lo && l < hi).length;
    histBins.push({ range: `${Math.round(lo)}-${Math.round(hi)}ms`, count });
  }

  // Latency by category
  const catLatency = Object.entries(by_category || {}).map(([name, _]) => {
    const catPairs = (pairs || []).filter((p) => p.category === name);
    const avg = catPairs.reduce((s, p) => s + p.latency_ms, 0) / Math.max(catPairs.length, 1);
    const max = Math.max(...catPairs.map((p) => p.latency_ms), 0);
    const min = Math.min(...catPairs.map((p) => p.latency_ms), 0);
    return { name, avg: Math.round(avg), max: Math.round(max), min: Math.round(min) };
  });

  // Token usage by direction
  const tokenByDir = Object.entries(by_direction || {}).map(([name, _]) => {
    const dirPairs = (pairs || []).filter((p) => `${p.src_lang}->${p.tgt_lang}` === name);
    const totalTokens = dirPairs.reduce((s, p) => s + (p.token_estimate || 0), 0);
    return { name, tokens: totalTokens, pairs: dirPairs.length };
  });

  // Cost by direction
  const costByDir = Object.entries(by_direction || {}).map(([name, _], i) => {
    const dirPairs = (pairs || []).filter((p) => `${p.src_lang}->${p.tgt_lang}` === name);
    const cost = dirPairs.length * (total_cost_estimate / Math.max((pairs || []).length, 1));
    return { name, value: +cost.toFixed(5) };
  });

  return (
    <div>
      {/* Summary cards */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <MetricCard label="Avg Latency" value={avg_latency_ms} metric="latency" subtitle="milliseconds" />
        <MetricCard label="Total Cost" value={total_cost_estimate} metric="cost" subtitle="USD estimate" />
        <MetricCard
          label="Pairs Tested"
          value={(pairs || []).length}
          metric="count"
          subtitle="golden pairs"
        />
        <MetricCard
          label="Total Tokens"
          value={(pairs || []).reduce((s, p) => s + (p.token_estimate || 0), 0)}
          metric="tokens"
          subtitle="estimated"
        />
      </div>

      {/* Latency distribution */}
      <h3 style={{ fontSize: 14, marginBottom: 8 }}>Latency Distribution</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={histBins} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
          <XAxis dataKey="range" tick={{ fill: "#e9eef5", fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
          <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
          <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
          <Bar dataKey="count" fill="#2a6bff" radius={[4, 4, 0, 0]} name="Pairs" />
        </BarChart>
      </ResponsiveContainer>

      {/* Latency by category */}
      {catLatency.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Latency by Category (ms)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={catLatency} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="name" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="avg" fill="#2a6bff" radius={[4, 4, 0, 0]} name="Avg" />
              <Bar dataKey="max" fill="#ef4444" radius={[4, 4, 0, 0]} name="Max" />
              <Bar dataKey="min" fill="#22c55e" radius={[4, 4, 0, 0]} name="Min" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Token usage */}
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginTop: 24 }}>
        <div style={{ flex: 1, minWidth: 300 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Token Usage by Direction</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={tokenByDir} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#202841" />
              <XAxis dataKey="name" tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <YAxis tick={{ fill: "#e9eef5", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
              <Bar dataKey="tokens" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Tokens" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ flex: 1, minWidth: 200 }}>
          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Cost by Direction</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={costByDir}
                cx="50%"
                cy="50%"
                outerRadius={70}
                paddingAngle={2}
                dataKey="value"
                label={({ name, value }) => `${name}: $${value.toFixed(4)}`}
              >
                {costByDir.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#121726", border: "1px solid #2a3558", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
