FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY data/raw ./data/raw

RUN mkdir -p data/processed data/index data/eval artifacts/runs artifacts/reports

EXPOSE 8000

CMD ["uvicorn", "driveiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]