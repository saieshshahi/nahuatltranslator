import { useState, useCallback } from "react";
import { postJSON, getJSON } from "../api";

export function useEvalData() {
  const [evalResult, setEvalResult] = useState(null);
  const [corpusStats, setCorpusStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runEvaluation = useCallback(async (options = {}) => {
    setError("");
    setLoading(true);
    try {
      const res = await postJSON("/evaluate", {
        directions: options.directions || [],
        categories: options.categories || [],
        max_pairs: options.max_pairs || 79,
      });
      if (res.error) {
        setError(res.error);
      } else {
        setEvalResult(res);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchCorpusStats = useCallback(async () => {
    try {
      const res = await getJSON("/corpus/stats");
      setCorpusStats(res);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await getJSON("/evaluate/history");
      setHistory(res || []);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  return {
    evalResult,
    corpusStats,
    history,
    loading,
    error,
    runEvaluation,
    fetchCorpusStats,
    fetchHistory,
  };
}
