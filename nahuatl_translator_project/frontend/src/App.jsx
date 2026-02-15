import React, { useMemo, useState } from "react";
import { postJSON, postForm, API_BASE } from "./api";

const VARIETIES = ["Unknown", "Central", "Huasteca", "Pipil", "Other"];

function Header() {
  return (
    <div className="card">
      <h1 style={{ marginTop: 0 }}>Nahuatl Translator & Manuscript Transcriber</h1>
      <p style={{ marginTop: 6, opacity: 0.9 }}>
        Translate text and transcribe manuscript pages (image/PDF) using the configured backend.
      </p>
      <p style={{ marginTop: 6, opacity: 0.9 }}>
        Backend API: <code>{API_BASE}</code>
      </p>
    </div>
  );
}

function TranslateTab() {
  const [text, setText] = useState("");
  const [src, setSrc] = useState("en");
  const [tgt, setTgt] = useState("nah");
  const [variety, setVariety] = useState("Unknown");
  const [variants, setVariants] = useState(3);
  const [temperature, setTemperature] = useState(0.6);
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");

  async function run() {
    setErr("");
    setLoading(true);
    setOut(null);
    try {
      const res = await postJSON("/translate", {
        text,
        src,
        tgt,
        variety,
        variants: Number(variants),
        temperature: Number(temperature),
      });
      setOut(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Translation</h2>

      <div className="row">
        <div>
          <label>Source</label>
          <select value={src} onChange={(e) => setSrc(e.target.value)}>
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="nah">Nahuatl</option>
          </select>
        </div>

        <div>
          <label>Target</label>
          <select value={tgt} onChange={(e) => setTgt(e.target.value)}>
            <option value="nah">Nahuatl</option>
            <option value="en">English</option>
            <option value="es">Spanish</option>
          </select>
        </div>

        <div>
          <label>Variety / dialect hint</label>
          <select value={variety} onChange={(e) => setVariety(e.target.value)}>
            {VARIETIES.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="row">
        <div>
          <label>Variants (1–5)</label>
          <input
            type="number"
            min="1"
            max="5"
            value={variants}
            onChange={(e) => setVariants(e.target.value)}
          />
          <small>Returns multiple candidate translations for comparison.</small>
        </div>

        <div>
          <label>Temperature (0–2)</label>
          <input
            type="number"
            min="0"
            max="2"
            step="0.1"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
          />
          <small>Higher = more variation.</small>
        </div>

        <div style={{ display: "flex", alignItems: "end", gap: 10 }}>
          <button onClick={run} disabled={loading || !text.trim()}>
            {loading ? "Translating…" : "Translate"}
          </button>
          <button
            className="secondary"
            onClick={() => {
              setText("");
              setOut(null);
              setErr("");
            }}
          >
            Clear
          </button>
        </div>
      </div>

      <label>Input text</label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste text…"
      />

      {err && <p style={{ color: "#ff7a7a" }}>Error: {err}</p>}

      {out && out.variants && (
        <div style={{ marginTop: 12 }}>
          <label>Variants</label>
          {out.variants.map((t, i) => (
            <pre key={i}>
              <b>#{i + 1}</b>
              {"\n"}
              {t}
            </pre>
          ))}
        </div>
      )}
    </div>
  );
}

function TranscribeTab() {
  const [file, setFile] = useState(null);
  const [languageHint, setLanguageHint] = useState("es");
  const [alphabetHint, setAlphabetHint] = useState("");
  const [page, setPage] = useState(1);
  const [profile, setProfile] = useState(true);
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");

  async function run() {
    setErr("");
    setOut(null);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("language_hint", languageHint);
      form.append("alphabet_hint", alphabetHint);
      form.append("page_num", String(page));
      form.append("handwriting_profile", profile ? "true" : "false");
      const res = await postForm("/transcribe", form);
      setOut(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Manuscript Transcription</h2>

      <div className="row">
        <div>
          <label>Upload image or PDF</label>
          <input
            type="file"
            accept="image/*,application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <small>For PDFs, choose a page number (1-indexed).</small>
        </div>

        <div>
          <label>PDF page</label>
          <input
            type="number"
            min="1"
            value={page}
            onChange={(e) => setPage(e.target.value)}
          />
        </div>

        <div>
          <label>Language hint</label>
          <select value={languageHint} onChange={(e) => setLanguageHint(e.target.value)}>
            <option value="es">Spanish</option>
            <option value="nah">Nahuatl</option>
            <option value="en">English</option>
          </select>
        </div>
      </div>

      <label>Alphabet / orthography hint (optional)</label>
      <textarea
        value={alphabetHint}
        onChange={(e) => setAlphabetHint(e.target.value)}
        placeholder="Paste orthography rules here (optional)…"
      />

      <div className="row">
        <div>
          <label>Handwriting profile</label>
          <select
            value={profile ? "yes" : "no"}
            onChange={(e) => setProfile(e.target.value === "yes")}
          >
            <option value="yes">Generate profile JSON</option>
            <option value="no">Skip</option>
          </select>
        </div>

        <div style={{ display: "flex", alignItems: "end" }}>
          <button onClick={run} disabled={loading || !file}>
            {loading ? "Transcribing…" : "Transcribe"}
          </button>
        </div>
      </div>

      {err && <p style={{ color: "#ff7a7a" }}>Error: {err}</p>}

      {out && (
        <div style={{ marginTop: 12 }}>
          <label>Transcription</label>
          <pre>{out.transcription}</pre>

          {out.handwriting_profile && (
            <>
              <label>Handwriting profile</label>
              <pre>{out.handwriting_profile}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ExtractTab() {
  const [text, setText] = useState("");
  const [instruction, setInstruction] = useState("Extract people, places, dates, and key events.");
  const [schema, setSchema] = useState("");
  const [loading, setLoading] = useState(false);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");

  async function run() {
    setErr("");
    setOut(null);
    setLoading(true);
    try {
      const res = await postJSON("/extract", { text, instruction, schema_hint: schema });
      setOut(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Structured Extraction</h2>

      <div className="row">
        <div>
          <label>Instruction</label>
          <input value={instruction} onChange={(e) => setInstruction(e.target.value)} />
        </div>
        <div>
          <label>Schema hint (optional)</label>
          <input
            value={schema}
            onChange={(e) => setSchema(e.target.value)}
            placeholder='e.g., {"people":[],"dates":[]}'
          />
        </div>
      </div>

      <label>Text</label>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Paste transcription or document text…"
      />

      <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
        <button onClick={run} disabled={loading || !text.trim()}>
          {loading ? "Extracting…" : "Extract"}
        </button>
        <button
          className="secondary"
          onClick={() => {
            setText("");
            setOut(null);
            setErr("");
          }}
        >
          Clear
        </button>
      </div>

      {err && <p style={{ color: "#ff7a7a" }}>Error: {err}</p>}

      {out && (
        <div style={{ marginTop: 12 }}>
          <label>Output</label>
          <pre>{out.output ?? JSON.stringify(out, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const tabs = useMemo(
    () => [
      { id: "translate", label: "Translate", comp: <TranslateTab /> },
      { id: "transcribe", label: "Transcribe", comp: <TranscribeTab /> },
      { id: "extract", label: "Extract", comp: <ExtractTab /> },
    ],
    []
  );

  const [active, setActive] = useState("translate");
  const current = tabs.find((t) => t.id === active)?.comp;

  return (
    <div className="container">
      <Header />
      <div className="tabs">
        {tabs.map((t) => (
          <div
            key={t.id}
            className={`tab ${active === t.id ? "active" : ""}`}
            onClick={() => setActive(t.id)}
          >
            {t.label}
          </div>
        ))}
      </div>
      {current}
    </div>
  );
}
