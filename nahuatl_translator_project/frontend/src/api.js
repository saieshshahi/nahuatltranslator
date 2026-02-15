export const API_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000")
  .replace(/\/+$/, "");

async function request(path, options) {
  const url = `${API_BASE}${path}`;
  const r = await fetch(url, options);
  const text = await r.text();

  if (!r.ok) {
    throw new Error(text || `HTTP ${r.status}`);
  }

  return text ? JSON.parse(text) : null;
}

async function requestWithWakeRetry(path, options) {
  try {
    return await request(path, options);
  } catch (e) {
    const msg = String(e?.message || "");
    const isLikelySleeping =
      msg.includes("502") ||
      msg.includes("503") ||
      msg.includes("504") ||
      msg.toLowerCase().includes("failed to fetch");

    if (!isLikelySleeping) throw e;

    // One retry after short delay (Render free tier often needs a moment to wake)
    await new Promise((res) => setTimeout(res, 2500));
    return await request(path, options);
  }
}

export async function postJSON(path, body) {
  return await requestWithWakeRetry(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function postForm(path, form) {
  return await requestWithWakeRetry(path, {
    method: "POST",
    body: form,
  });
}
