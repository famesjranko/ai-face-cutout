FROM python:3.12-slim

# System deps for OpenCV (ultralytics pulls full opencv-python which needs libGL)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (big layer, cache it)
RUN pip install --no-cache-dir \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code
COPY weights/ weights/
COPY server/ server/
COPY static/ static/

EXPOSE 8000

CMD ["python", "-m", "server.run"]
