FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Persistent data lives on a mounted volume at /data
# agent.py and db.py read DATA_DIR to find state/ and arbitrage.db
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

# Create local fallback dirs (used when no volume is mounted, e.g. local dev)
RUN mkdir -p /data state

EXPOSE 8000

CMD ["python", "agent.py"]
