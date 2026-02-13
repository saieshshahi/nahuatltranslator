# Nahuatl Translator + Document OCR (English/Spanish ↔ Nahuatl)

This project turns an Excel parallel corpus (English–Nahuatl) into train/valid/test splits, fine-tunes a seq2seq model, and serves a clean web UI that also supports **document upload + OCR**.

## What’s inside
- `data/english_to_nahuatl_parallel.xlsx` (your uploaded dataset)
- `scripts/prepare_data.py` → cleans + creates JSONL splits
- `scripts/train_accelerate.py` → fine-tunes a model (CPU/GPU)
- `scripts/eval.py` → BLEU/chrF on test
- `webapp/app.py` → Gradio web UI (Translate + Upload/OCR + Manual extraction)

## Recommended runtime
- **GPU strongly recommended** for training. CPU works but is slow.

## Quickstart (Colab or local)

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Prepare data
```bash
python scripts/prepare_data.py --excel data/english_to_nahuatl_parallel.xlsx --out data/splits --max_len 80
```

By default, this creates **bidirectional** training records so your model learns both:
* English → Nahuatl
* Nahuatl → English

### 3) Train (fast demo)
This trains quickly by using a subset and 1 epoch.
```bash
python scripts/train_accelerate.py \
  --model t5-small \
  --data_dir data/splits \
  --out_dir runs/t5_small_demo \
  --epochs 1 \
  --train_examples 2000
```

### 4) Train (better quality)
```bash
python scripts/train_accelerate.py \
  --model google/mt5-small \
  --data_dir data/splits \
  --out_dir runs/mt5_small \
  --epochs 3 \
  --batch_size 16
```

### 5) Evaluate
```bash
python scripts/eval.py --model_dir runs/mt5_small --data_dir data/splits
```

### 6) Run the web app
```bash
export MODEL_DIR=runs/mt5_small
python webapp/app.py
```
Open http://127.0.0.1:7860

Optional (for higher-quality fallback + instruction-following extraction):
```bash
export OPENAI_API_KEY=...   # optional
```

You can also set it via a `.env` file (recommended for Docker):

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Optional model overrides:

```bash
export OPENAI_TRANSLATE_MODEL=gpt-4o-mini
export OPENAI_EXTRACT_MODEL=gpt-4o-mini
```

## Notes on “varieties / dialects”
Your Excel currently contains one Nahuatl variety (from a single source). The pipeline supports a `variety` label column.
- If your input has no variety info, `prepare_data.py` will set `variety="Unknown"`.
- If you later add datasets from multiple varieties, include a `variety` column (e.g., `nhi`, `nhw`, etc.) and retrain.

## Troubleshooting
- If training is slow and you see `GPU: False`, enable GPU (Colab: Runtime → Change runtime type → GPU).
- First run will download model weights (can be hundreds of MB to 1.2GB).

---

## Run with Docker (recommended: no dependency conflicts)

### 1) Install Docker
- Windows/macOS: Docker Desktop
- Linux: Docker Engine + Docker Compose

### 2) Build + run the web app
From this folder:

If you want OpenAI fallback / extraction, create a `.env` file first:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

```bash
docker compose up --build
```

If you want OpenAI fallback, create a `.env` file first:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Open: http://localhost:7860

### 3) Prepare data / train / evaluate (inside Docker)

```bash
docker compose run --rm nahuatl python scripts/prepare_data.py --excel data/english_to_nahuatl_parallel.xlsx --out data/splits
docker compose run --rm nahuatl python scripts/train_accelerate.py --model google/mt5-small --data_dir data/splits --out_dir runs/mt5_small --epochs 3
docker compose run --rm nahuatl python scripts/eval.py --model_dir runs/mt5_small --data_dir data/splits
```

Then run again:

```bash
docker compose up --build
```

---

## GitHub Pages + Render deployment (Option 1)

This repo includes a **static frontend** (GitHub Pages) and a **FastAPI backend** (Render).

- **Frontend:** `frontend/` (React/Vite)
- **Backend API:** `backend/main.py` (FastAPI, Docker)

See **DEPLOY_GITHUB_PAGES_RENDER.md** for the step-by-step.

### Local dev
Backend:
```bash
export OPENAI_API_KEY=... 
pip install -r requirements.txt -r backend/requirements_api.txt
uvicorn backend.main:app --reload --port 8000
```
Frontend:
```bash
cd frontend
npm install
VITE_API_BASE=http://localhost:8000 npm run dev
```
