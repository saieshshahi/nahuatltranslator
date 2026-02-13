FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860
ENV PORT=7860
# MODEL_DIR can point to a fine-tuned model folder (mounted volume or baked into the image)
ENV MODEL_DIR=runs/mt5_small
CMD ["python", "webapp/app.py"]
