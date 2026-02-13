export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export async function postJSON(path, body){
  const r = await fetch(`${API_BASE}${path}`, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  })
  const text = await r.text()
  if(!r.ok) throw new Error(text || `HTTP ${r.status}`)
  return JSON.parse(text)
}

export async function postForm(path, form){
  const r = await fetch(`${API_BASE}${path}`, { method:'POST', body: form })
  const text = await r.text()
  if(!r.ok) throw new Error(text || `HTTP ${r.status}`)
  return JSON.parse(text)
}
