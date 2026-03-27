# CricketBrain AI — Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies (for LightGBM, XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data dirs
RUN mkdir -p data/raw data/cleaned data/features data/models data/shap

# Expose ports
EXPOSE 8000 8501

# Default: start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
