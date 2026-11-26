FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf XDG_CACHE_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
# INSTALL_ML=false builds a small image for the fake backend (CI, local
# Kubernetes tests). The real image adds CPU-only PyTorch and the models' libraries.
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then \
      pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-ml.txt; \
    fi
COPY analyzer ./analyzer
COPY service ./service
RUN useradd --system --uid 10001 app && mkdir -p /models && chown app /models
USER app
EXPOSE 9100
CMD ["python", "-m", "service.worker"]
