# backend/Dockerfile
FROM python:3.10-slim

# No .pyc + unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System libs needed for PyMuPDF + lxml + chroma deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Chroma + Assets directories
ENV CHROMA_PERSIST_DIR=/app/chroma_db
ENV ASSETS_DIR=/app/assets

RUN mkdir -p $CHROMA_PERSIST_DIR $ASSETS_DIR

# Expose FastAPI port
ENV PORT=8000
EXPOSE 8000

# IMPORTANT FIX → run backend/app.py correctly
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
