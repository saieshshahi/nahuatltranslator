FROM python:3.11-slim

# System deps for OCR (tesseract) and PyMuPDF runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
COPY backend/requirements_api.txt /app/requirements_api.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements_api.txt

# Copy source
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
