import React, { useState } from "react";
import { useEvalData } from "./metrics/useEvalData";
import TranslationQuality from "./metrics/TranslationQuality";
import LLMPerformance from "./metrics/LLMPerformance";
import CorpusData from "./metrics/CorpusData";
import FineTuning from "./metrics/FineTuning";
import Trends from "./metrics/Trends";

const SUB_TABS = [
  { id: "quality", label: "Translation Quality" },
  { id: "llm", label: "LLM Performance" },
  { id: "corpus", label: "Corpus & Data" },
  { id: "finetune", label: "Fine-tuning" },
  { id: "trends", label: "Trends" },
];

export default function MetricsTab() {
  const [subTab, setSubTab] = useState("quality");
  const {
    evalResult,
    corpusStats,
    history,
    loading,
    error,
    runEvaluation,
    fetchCorpusStats,
    fetchHistory,
  } = useEvalData();

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
        <h2 style={{ marginTop: 0, marginBottom: 0 }}>Evaluation Metrics</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => runEvaluation()}
            disabled={loading}
            style={{ width: "auto", padding: "8px 20px" }}
          >
            {loading ? "Evaluating..." : "Run Evaluation"}
          </button>
          {evalResult?.cached && (
            <span style={{ fontSize: 11, opacity: 0.6, alignSelf: "center" }}>
              (cached)
            </span>
          )}
        </div>
      </div>

      {error && <p style={{ color: "#ff7a7a", marginTop: 8 }}>Error: {error}</p>}

      {loading && (
        <div style={{ marginTop: 12, padding: 20, textAlign: "center", opacity: 0.7 }}>
          <p>Running evaluation on golden pairs... This takes 30-60 seconds.</p>
          <div style={{
            width: 200,
            height: 4,
            background: "#202841",
            borderRadius: 2,
            margin: "12px auto",
            overflow: "hidden",
          }}>
            <div style={{
              width: "60%",
              height: "100%",
              background: "#2a6bff",
              borderRadius: 2,
              animation: "pulse 1.5s ease-in-out infinite",
            }} />
          </div>
        </div>
      )}

      {/* Sub-tabs */}
      <div style={{ display: "flex", gap: 6, marginTop: 16, marginBottom: 16, flexWrap: "wrap" }}>
        {SUB_TABS.map((t) => (
          <div
            key={t.id}
            onClick={() => setSubTab(t.id)}
            style={{
              padding: "6px 14px",
              borderRadius: 8,
              background: subTab === t.id ? "#1b2b57" : "#11182b",
              border: `1px solid ${subTab === t.id ? "#2a6bff" : "#202841"}`,
              cursor: "pointer",
              fontSize: 12,
              fontWeight: subTab === t.id ? 600 : 400,
              transition: "all 0.15s",
            }}
          >
            {t.label}
          </div>
        ))}
      </div>

      {/* Sub-tab content */}
      {subTab === "quality" && <TranslationQuality evalResult={evalResult} />}
      {subTab === "llm" && <LLMPerformance evalResult={evalResult} />}
      {subTab === "corpus" && <CorpusData corpusStats={corpusStats} fetchCorpusStats={fetchCorpusStats} />}
      {subTab === "finetune" && <FineTuning />}
      {subTab === "trends" && <Trends history={history} fetchHistory={fetchHistory} />}
    </div>
  );
}
