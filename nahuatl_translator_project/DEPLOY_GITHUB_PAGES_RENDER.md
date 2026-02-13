# Deploy: GitHub Pages (frontend) + Render (backend)

This repo is set up for **Option 1**:
- **GitHub Pages** hosts the static website (React).
- **Render** hosts the API (FastAPI in Docker) that talks to OpenAI and does OCR/PDF page rendering.

You will do **two** deployments:
1) Render backend → get a public API URL
2) GitHub Pages frontend → point it at that API URL

---

## 0) Prerequisites
- A GitHub account (to host the repo + Pages)
- A Render account (free is fine)
- An OpenAI API key

---

## 1) Put this code on GitHub
1. Create a new GitHub repo (example: `nahuatl-tools`).
2. Upload/push this project to the repo (main branch).

---

## 2) Deploy the backend on Render

### A) One‑click blueprint deploy (easiest)
1. In Render, click **New +** → **Blueprint**
2. Select your GitHub repo
3. Render detects `render.yaml` automatically
4. Set the environment variable:
   - `OPENAI_API_KEY` = your key
5. Click **Apply** / **Deploy**

When it finishes, copy your Render service URL, e.g.:
- `https://nahuatl-backend.onrender.com`

### B) Important: set CORS correctly
By default the backend allows `*`.
If you want to lock it down, set `CORS_ORIGINS` on Render to your Pages URL:
- `https://YOUR_USERNAME.github.io`

---

## 3) Deploy the frontend on GitHub Pages

### A) Add the backend URL to GitHub Actions Secrets
1. In your GitHub repo: **Settings** → **Secrets and variables** → **Actions**
2. Add a secret:
   - Name: `VITE_API_BASE`
   - Value: your Render URL (e.g. `https://nahuatl-backend.onrender.com`)

### B) Enable Pages from GitHub Actions
1. Repo **Settings** → **Pages**
2. Under **Build and deployment**:
   - Source: **GitHub Actions**

### C) Update the workflow to use the secret
The workflow is already installed. It reads `VITE_API_BASE` automatically.
If you changed anything, push to `main` to trigger it.

After the workflow completes, your site will be available at:
- `https://YOUR_USERNAME.github.io/REPO_NAME/`

---

## 4) Local development (optional)

### Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r backend/requirements_api.txt
export OPENAI_API_KEY=...
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
VITE_API_BASE=http://localhost:8000 npm run dev
```
Open the dev site at the Vite URL (usually `http://localhost:5173`).

---

## 5) API endpoints
- `GET /health`
- `POST /api/translate` JSON: `{ text, src, tgt, variety, variants, temperature }`
- `POST /api/transcribe` multipart: `file`, plus fields `language_hint`, `alphabet_hint`, `page_num`, `handwriting_profile`
- `POST /api/extract` JSON: `{ text, instruction, schema_hint }`
- `POST /api/ocr` multipart: `file`, field `lang_hint`

---

## Troubleshooting

### “CORS error” in the browser
Set `CORS_ORIGINS` on Render to:
- `https://YOUR_USERNAME.github.io`

### “OpenAI key not set”
Set `OPENAI_API_KEY` in Render → Environment.

### PDFs fail to render
If the PDF is password-protected or extremely large, try exporting a single page or uploading a screenshot instead.
