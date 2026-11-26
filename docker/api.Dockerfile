FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY analyzer ./analyzer
COPY service ./service
RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000
CMD ["uvicorn", "service.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
